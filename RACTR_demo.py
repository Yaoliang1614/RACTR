import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from skimage.metrics import structural_similarity
from sympy.abc import epsilon
from torch import nn, optim
import time

class Config:

    ## default settings
    SEED =3407
    TRAIN_MODE = "tube"  # "tube" 或 "point"
    TUBE_BATCH_SIZE = 8192  # 每步采样的空间位置数量
    #POINT_BATCH_SIZE = 8192  # 每步采样的有效观测点数量
    LOG_EVERY = 100
    EVAL_CHUNK_PIXELS = 1024
    ITERATIONS = 4000
    RANK_R = 256
    SHOW_RESULT = False
    ## network settings
    SPATIAL_LAYERS = 6
    SPATIAL_HIDDEN_DIM = 256
    SPECTRAL_LAYERS = 6
    SPECTRAL_HIDDEN_DIM = 64
    NUM_FREQS = 128

    # TEST_FILE =r"your path of test file"
    # Our demo is with "Houston_0.6NR_0.1SR_1.mat" or "DCMall_gr_0.2.mat" as the example

    TEST_FILE="./Houston_0.6NR_0.1SR_1.mat"
    FOURIER_SCALE = 1# belong to the range of (1,15).
    LOSS_FAMILY = "gm"  # "gm","l2"
    LOSS_PARAM = 0.1# Geman —> alpha:[0.05,0.1,0.5] for better;  L2 —> None。
    LR = 1e-3    #For Sparse denoising-> 0.001
    NEGATIVE_SLOPE = 0.0   # [0.0 or 0.1]


    # TEST_FILE="./DCMall_gr_0.2.mat"
    # FOURIER_SCALE = 3 # belong to the range of (1,15).
    # LOSS_FAMILY = "l2"  # "gm","l2"
    # LOSS_PARAM = None # Geman —> alpha:[0.05,0.1,0.5] for better;  L2 —> None。
    # LR = 1e-4    #For Gaussian denoising-> 0.0001
    # NEGATIVE_SLOPE = 0.1   # [0.0 or 0.1]

def validate_config(cfg):
    if cfg.TRAIN_MODE not in ("tube", "point"):
        raise ValueError("TRAIN_MODE 只支持 tube 或 point。")
    batch_name = (
        "TUBE_BATCH_SIZE" if cfg.TRAIN_MODE == "tube"
        else "POINT_BATCH_SIZE"
    )
    for name in (
        "RANK_R", "SPATIAL_HIDDEN_DIM", "SPECTRAL_HIDDEN_DIM",
        "NUM_FREQS", "ITERATIONS", "LOG_EVERY", "EVAL_CHUNK_PIXELS",
        batch_name,
    ):
        value = getattr(cfg, name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须为正整数。")
    for name in ("SPATIAL_LAYERS", "SPECTRAL_LAYERS"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} 至少为 2，包含输出层。")
    if not np.isfinite(cfg.LR) or cfg.LR <= 0:
        raise ValueError("LR 必须为有限的正数。")
    if not np.isfinite(cfg.NEGATIVE_SLOPE) or cfg.NEGATIVE_SLOPE < 0:
        raise ValueError("NEGATIVE_SLOPE 必须为有限的非负数。")
    if not np.isfinite(cfg.FOURIER_SCALE) or cfg.FOURIER_SCALE <= 0:
        raise ValueError("FOURIER_SCALE 必须为有限的正数。")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RobustLoss(nn.Module):
    def __init__(self, family, param=None, epsilon=1e-6):
        super().__init__()
        self.family = family.lower()
        if self.family not in ("gm", "l2"):
            raise ValueError(f"不支持的损失类型：{family}")
        if self.family != "l2":
            if param is None or not np.isfinite(param) or param <= 0:
                raise ValueError("非凸损失的 p/alpha 必须为有限的正数。")
        self.param = None if self.family == "l2" else float(param)
        self.epsilon = epsilon

    def forward(self, pred, target):
        # 保留上传版本的数值公式，包括所有分支中的 epsilon。
        e = torch.abs(pred - target)+self.epsilon
        if self.family == "gm":
            penalty = self.param * e / (e + self.param)
        elif self.family == "l2":
            penalty = (pred - target) ** 2
        return penalty.mean()


class FactorMLP(nn.Module):
    def __init__(self, in_dim, rank, hidden, layers, num_freqs, scale, slope):
        super().__init__()
        # 固定 Fourier 矩阵，不属于可训练参数。
        self.register_buffer("B", torch.randn(in_dim, num_freqs) * scale)
        modules = [
            nn.Linear(2 * num_freqs, hidden),
            nn.LeakyReLU(negative_slope=slope),
        ]
        for _ in range(layers - 2):
            modules.extend([
                nn.Linear(hidden, hidden),
                nn.LeakyReLU(negative_slope=slope),
            ])
        modules.append(nn.Linear(hidden, rank))
        self.mlp = nn.Sequential(*modules)

    def forward(self, coords):
        projected = 2.0 * np.pi * (coords @ self.B)
        features = torch.cat(
            (torch.sin(projected), torch.cos(projected)), dim=-1
        )
        return self.mlp(features)


class RACTR(nn.Module):
    def __init__(self, cfg, scale):
        super().__init__()
        # [2, 1] grouping: model (x, y) jointly and z separately.
        self.spatial_branches = nn.ModuleList([
            FactorMLP(
                2, cfg.RANK_R, cfg.SPATIAL_HIDDEN_DIM,
                cfg.SPATIAL_LAYERS, cfg.NUM_FREQS, scale,
                cfg.NEGATIVE_SLOPE,
            )
        ])
        self.spectral_branch = FactorMLP(
            1, cfg.RANK_R, cfg.SPECTRAL_HIDDEN_DIM,
            cfg.SPECTRAL_LAYERS, cfg.NUM_FREQS, scale,
            cfg.NEGATIVE_SLOPE,
        )

    def spatial_features(self, xy):
        return self.spatial_branches[0](xy)

    def forward(self, coords):
        spatial = self.spatial_features(coords[:, :2])
        spectral = self.spectral_branch(coords[:, 2:3])
        return (spatial * spectral).sum(dim=-1, keepdim=True)

    def predict_tubes(self, xy, z):
        return self.spatial_features(xy) @ self.spectral_branch(z).T


def load_dataset(path):
    data = sio.loadmat(path)
    clean = data["X"].astype(np.float32)
    if "Xmiss" in data:
        observed = data["Xmiss"].astype(np.float32)
    elif "Xn" in data:
        observed = data["Xn"].astype(np.float32)
    else:
        raise KeyError("缺少观测变量 Xmiss 或 Xn。")
    mask = (
        data["sampling_mask"].astype(np.float32) > 0.5
        if "sampling_mask" in data else np.ones_like(observed, dtype=bool)
    )
    if observed.ndim != 3 or min(observed.shape) < 1:
        raise ValueError("输入必须为非空的 [高度, 宽度, 波段数] 张量。")
    if clean.shape != observed.shape or mask.shape != observed.shape:
        raise ValueError("X、观测和 sampling_mask 的形状必须一致。")
    if not np.any(mask):
        raise ValueError("Mask 中没有有效观测。")
    # 观测按文件原样加载。
    return clean, observed, mask


def make_coordinates(shape, device):
    h, w, bands = shape
    xy = torch.stack(
        torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w),
            indexing="ij",
        ), dim=-1,
    ).reshape(h * w, 2).to(device)
    z = torch.linspace(-1, 1, bands).reshape(bands, 1).to(device)
    return xy, z


def calculate_mpsnr(gt, recon):
    gt = np.clip(gt, 0.0, 1.0)
    recon = np.clip(recon, 0.0, 1.0)
    values = []
    for band in range(gt.shape[2]):
        mse = np.mean((gt[:, :, band] - recon[:, :, band]) ** 2)
        values.append(10 * np.log10(1.0 / mse) if mse > 0 else 100.0)
    return float(np.mean(values))


def calculate_mssim(gt, recon):
    gt = np.clip(gt, 0.0, 1.0)
    recon = np.clip(recon, 0.0, 1.0)
    return float(np.mean([
        structural_similarity(
            gt[:, :, band], recon[:, :, band], data_range=1.0
        )
        for band in range(gt.shape[2])
    ]))


@torch.no_grad()
def reconstruct(model, xy, z, shape, chunk_pixels):
    was_training = model.training
    model.eval()
    try:
        h, w, bands = shape
        output = np.empty((h * w, bands), dtype=np.float32)
        spectral = model.spectral_branch(z)
        for begin in range(0, h * w, chunk_pixels):
            end = min(begin + chunk_pixels, h * w)
            spatial = model.spatial_features(xy[begin:end])
            output[begin:end] = (spatial @ spectral.T).cpu().numpy()
        return output.reshape(shape)
    finally:
        model.train(was_training)


def masked_tube_loss(pred, target, mask, criterion, observed_fraction):
    if observed_fraction == 1.0:
        return criterion(pred, target)
    pred_valid = pred[mask]
    if pred_valid.numel() == 0:
        return pred.sum() * 0.0
    # 全局观测比例归一化，保持原平均保真项的期望权重。
    return (
        criterion(pred_valid, target[mask])
        * (pred_valid.numel() / pred.numel()) / observed_fraction
    )


def train_session(clean_np, observed_np, mask_np, params, device, cfg):
    set_seed(cfg.SEED)
    h, w, bands = observed_np.shape
    pixels = h * w
    observed = torch.from_numpy(observed_np).reshape(pixels, bands).to(device)
    mask = torch.from_numpy(mask_np).reshape(pixels, bands).to(device)
    valid_np = np.flatnonzero(mask_np.reshape(-1))
    observed_fraction = valid_np.size / observed_np.size
    valid = (
        torch.from_numpy(valid_np).long().to(device)
        if cfg.TRAIN_MODE == "point" else None
    )
    xy, z = make_coordinates(observed_np.shape, device)
    model = RACTR(cfg, params["SCALE"]).to(device)
    criterion = RobustLoss(params["FAMILY"], params.get("PARAM"))
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"MODE_SPLITS=[2, 1]; TRAIN_MODE={cfg.TRAIN_MODE}; "
        f"可训练参数={parameter_count:,}"
    )

    # 开始统计本组配置的训练耗时和峰值显存
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()

    model.train()
    for step in range(1, cfg.ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)
        if cfg.TRAIN_MODE == "tube":
            indices = torch.randint(
                0, pixels, (cfg.TUBE_BATCH_SIZE,), device=device
            )
            pred = model.predict_tubes(xy[indices], z)
            loss = masked_tube_loss(
                pred, observed[indices], mask[indices],
                criterion, observed_fraction,
            )
        else:
            indices = valid[torch.randint(
                0, valid.numel(), (cfg.POINT_BATCH_SIZE,), device=device
            )]
            coords = torch.cat(
                (xy[indices // bands], z[indices % bands]), dim=-1
            )
            loss = criterion(model(coords), observed.reshape(-1, 1)[indices])
        loss.backward()
        optimizer.step()


        if step % cfg.LOG_EVERY == 0:
            current = reconstruct(
                model, xy, z, observed_np.shape, cfg.EVAL_CHUNK_PIXELS
            )
            psnr = calculate_mpsnr(clean_np, current)
            print(
                f"Step {step}/{cfg.ITERATIONS}, MPSNR={psnr:.4f} dB"
            )
            del current
        elif step == 1:
            print(f"Step {step}/{cfg.ITERATIONS}, loss={loss.item():.8f}")

    # 中间 PSNR 用于日志，返回最后一次迭代的重建。
    recon = reconstruct(
        model, xy, z, observed_np.shape, cfg.EVAL_CHUNK_PIXELS
    )
    return {
        "Xhat": np.clip(recon, 0.0, 1.0),
        "psnr": calculate_mpsnr(clean_np, recon),
        "ssim": calculate_mssim(clean_np, recon),
        "params": params,
        "trainable_params": parameter_count,
    }


def show_result(clean, observed, recovered):
    def display(array):
        array = np.clip(array, 0.0, 1.0)
        return array if array.shape[2] == 3 else array[:, :, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, array, title in zip(
        axes, (clean, observed, recovered), ("Clean", "Observation", "Recovered")
    ):
        image = display(array)
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def main(cfg=Config):
    validate_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean, observed, mask = load_dataset(cfg.TEST_FILE)

    params = {"SCALE": float(cfg.FOURIER_SCALE), "FAMILY": cfg.LOSS_FAMILY}
    if cfg.LOSS_PARAM is not None:
        params["PARAM"] = float(cfg.LOSS_PARAM)
    print(f"设备={device}; 固定参数={params}")
    result = train_session(clean, observed, mask, params, device, cfg)

    print(f"\n最终 MPSNR={result['psnr']:.4f} dB; MSSIM={result['ssim']:.6f}")
    if cfg.SHOW_RESULT:
        show_result(clean, observed, result["Xhat"])
    return result


if __name__ == "__main__":
    main()


