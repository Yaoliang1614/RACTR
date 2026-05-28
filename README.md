# RACTR
### 💻 Implementation Details

#### 1. Hardware & Software Environment
All experiments and codes were implemented and evaluated on a desktop platform equipped with a consumer-grade GPU:
* **IDE**: PyCharm
* **GPU**: NVIDIA RTX 4060 Ti
* **VRAM**: 8 GB

#### 2. Hyperparameter Settings
The RACTR model is designed to be concise, requiring the adjustment of only two hyperparameters: `s` and `α`. For all evaluated tasks, the search space for these parameters is defined as follows:
* **Scale Parameter (`s`)**: {1, 5, 10, 15}
* **Weight Parameter (`α`)**: {0.05, 0.1, 0.5, 1, 5}

#### 3. Penalty Function Configuration
* **Main Configuration**: For the sake of conciseness and to comprehensively demonstrate the superiority of the RACTR model, the **Geman function** is utilized as the default penalty function in our primary experiments.
* **Extended Exploration**: Meanwhile, we also appropriately explore and compare the performance of other penalty functions within our code.

----------------------------------------------
### 📦 Data Loading Requirements

This repository contains two core tasks: **TRPCA (Tensor Denoising)** and **TC+TRPCA (Joint Tensor Completion and Denoising)**. Before running the algorithms, please ensure that your loaded dataset contains the corresponding variables described below:

#### 1. TRPCA Task (Pure Denoising)
For the standard tensor denoising task, you need to load both the fully observed noisy data and the clean data for evaluation.

| Variable | Type / Shape | Description & Meaning |
| :--- | :--- | :--- |
| `X` | Tensor / Matrix | **Ground Truth (Clean Data)**. Serves as the baseline reference and is used exclusively for calculating final evaluation metrics (e.g., MPSNR, MSSIM). |
| `Xn` | Tensor / Matrix | **Noisy Observation**. The actual input tensor for the algorithm, corrupted by an unknown noise distribution. |

#### 2. TC+TRPCA Joint Task (Completion + Denoising)
In the joint task, the observed data is not only corrupted by noise but also suffers from missing entries (undersampled), which requires the introduction of a masking mechanism.

| Variable | Type / Shape | Description & Meaning |
| :--- | :--- | :--- |
| `X` | Tensor / Matrix | **Ground Truth (Clean Data)**. The perfect reference used for metric evaluation. |
| `Xmiss` | Tensor / Matrix | **Undersampled & Noisy Observation**. The actual input for the algorithm, where some elements are known (but corrupted by noise) and others are completely missing. |
| `sampling_mask` | Boolean Tensor | **Sampling Mask**. A binary indicator tensor composed of 0s and 1s, exactly matching the dimensions of the data. `1` indicates an observed value exists at that position, while `0` indicates a missing value. |

> **⚠️ Note:**
> Please ensure that the spatial and spectral dimensions of `X`, `Xn` (or `Xmiss`), and the `sampling_mask` are **strictly identical**. Otherwise, dimension mismatch errors will occur during tensor operations.
-------------------------------------------------------------
## 📊 (Experimental Results)

This section presents the quantitative evaluation results of TRPCA (Tensor Robust Principal Component Analysis) and its combined task (TC+TRPCA) on different datasets under various noise and missing data conditions. The main evaluation metrics include Mean Peak Signal-to-Noise Ratio (MPSNR) and Mean Structural Similarity Index Measure (MSSIM).

### 1. TRPCA 

In the single denoising task, all experiments adopt `gm` (Geman-McClure) as FAMILY parameter。

| Dataset | SR | SCALE | PARAM | MPSNR (dB) | MSSIM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Carphone** | 0.1 | 5.0 | 0.10 | 42.58 | 0.9959 |
| | 0.2 | 10.0 | 0.05 | 40.40 | 0.9920 |
| | 0.3 | 5.0 | 0.10 | 37.26 | 0.9740 |
| **Akiyo** | 0.1 | 5.0 | 0.10 | 44.57 | 0.9963 |
| | 0.2 | 5.0 | 0.10 | 40.68 | 0.9926 |
| | 0.3 | 5.0 | 0.10 | 38.84 | 0.9878 |
| **Pavia** | 0.4 | 10.0 | 0.50 | 43.33 | 0.9949 |
| | 0.5 | 10.0 | 0.50 | 41.93 | 0.9931 |
| | 0.6 | 10.0 | 0.50 | 39.07 | 0.9872 |
| **HOuston** | 0.4 | 10.0 | 0.50 | 48.30 | 0.9977 |
| | 0.5 | 10.0 | 0.50 | 48.68 | 0.9967 |
| | 0.6 | 10.0 | 0.10 | 46.61 | 0.9967 |
| **Flowers** | 0.4 | 10.0 | 0.50 | 42.33 | 0.9835 |
| | 0.5 | 10.0 | 0.10 | 41.41 | 0.9806 |
| | 0.6 | 10.0 | 0.10 | 40.43 | 0.9774 |
| **Toys** | 0.4 | 10.0 | 0.10 | 42.26 | 0.9873 |
| | 0.5 | 10.0 | 0.10 | 41.53 | 0.9857 |
| | 0.6 | 10.0 | 0.10 | 39.50 | 0.9822 |

### 2. (TC+TRPCA)

The joint task tested the robustness of the algorithm under different ratios of Super-Resolution (SR) and Noise Reduction (NR). The FAMILY parameter was kept as `gm`。

| Dataset | (SR+NR) | SCALE | PARAM | MPSNR (dB) | MSSIM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DCmall** | 0.3SR + 0.1NR | 5.0 | 1.00 | 44.49 | 0.9956 |
| | 0.2SR + 0.3NR | 5.0 | 0.50 | 42.41 | 0.9924 |
| | 0.1SR + 0.5NR | 5.0 | 0.10 | 35.94 | 0.9718 |
| **Urban** | 0.3SR + 0.1NR | 10.0 | 5.00 | 43.11 | 0.9928 |
| | 0.2SR + 0.3NR | 10.0 | 0.50 | 39.65 | 0.9857 |
| | 0.1SR + 0.5NR | 5.0 | 0.10 | 32.44 | 0.9468 |

---

### 💡 Performance Analysis

Based on the above quantitative indicators, the following objective evaluation conclusions are drawn:

#### (Strengths)
* **Excellent Structure Preservation**: Under the vast majority of test conditions, the MSSIM remains stably **above 0.98**. This indicates that the algorithm perfectly preserves core structural information, such as textures and edges, during the denoising or joint reconstruction processes.
* **High Compatibility with Hyperspectral/Remote Sensing Data**: Under extremely high noise conditions (0.4~0.6), the algorithm still maintains an exceptionally high MPSNR of **46dB-48dB** on datasets like `HOuston`. The low-rank prior of the model demonstrates strong adaptability to the intrinsic spatial and spectral correlations of such data.
* **High Robustness Against Noise**: In the single denoising task, the performance exhibits a linear decay consistent with physical intuition as the noise intensity increases, without encountering any sudden model collapse.

#### (Limitations & Risks)
* **Bottlenecks in Extreme Conditions for Joint Tasks**: In the joint task (TC+TRPCA), when facing the dual pressure of **extreme information scarcity and high noise** (e.g., `0.1SR + 0.5NR`), the performance experiences a drastic drop (e.g., the MPSNR of the `Urban` dataset drops to 32.44 dB). Under these circumstances, the model is prone to falling into ill-posed solutions or generating over-smoothed results.
* **Parameter Sensitivity and Calibration Cost**: The algorithm is not completely parameter-free. Parameters such as `SCALE` require manual intervention when dealing with different data types and task difficulties (e.g., set to 5.0, but 10.0 for highly noisy remote sensing data). This introduces a certain degree of pre-calibration cost during engineering deployment.
