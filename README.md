# Robust and Anisotropic Continuous Tensor Recovery via Structural Topology and Nonconvex Residual Reweighting (RACTR)
### Our code will be available during these days (before November 20th). The following contexts are some details of our paper.

### 💻 Implementation Details

#### 1. Hardware & Software Environment
All experiments and codes were implemented and evaluated on a desktop platform equipped with a consumer-grade GPU:
* **IDE**: PyCharm
* **GPU**: NVIDIA RTX 4060 Ti
* **VRAM**: 8 GB
##### 1.1 **Neural networks** 


* HIDDEN_DIM_xy = 256
* HIDDEN_DIM_z = 64
* NUM_LAYERS = 6
* NUM_FREQS = 128
* R = 256
* LR = 0.001 (for sparse denoising)
* ITERATIONS = 4000
* BATCH_SIZE = 8192



#### 2. Hyperparameter Settings
The RACTR model is designed to be concise, requiring the adjustment of only two hyperparameters: `k` and `α`. For all evaluated tasks, the search space for these parameters is defined as follows:
* **Scale Parameter (`k`)**: {1, 5, 10, 15}
* **Nonconvex Parameter (`α`)**: {0.05, 0.1, 0.5, 1, 5}

#### 3. Penalty Function Configuration
* **Main Configuration**: For the sake of conciseness and to comprehensively demonstrate the superiority of the RACTR model, the **Geman function** is utilized as the default penalty function in our primary experiments.
* **Extended Exploration**: Meanwhile, we also appropriately explore and compare the performance of other penalty functions within our code.

----------------------------------------------
### 📦 Data Loading Requirements

This repository  shows multiple tasks: **Gaussian Denoising**, **Sparse Denoising (TRPCA)** and **Robust completion (Joint Tensor Completion and RPCA)**. Before running the algorithms, please ensure that your loaded dataset contains the corresponding variables described below:

#### 1. Gaussian Denoising task (Dense noising)
For the standard tensor denoising task, you need to load both the fully observed noisy data and the clean data for evaluation.
| penalty: L2 loss
| Variable | Type / Shape | Description & Meaning |
| :--- | :--- | :--- |
| `X` | Tensor / Matrix | **Ground Truth (Clean Data)**. Serves as the baseline reference and is used exclusively for calculating final evaluation metrics (e.g., MPSNR, MSSIM). |
| `Xn` | Tensor / Matrix | **Noisy Observation**. The actual input tensor for the algorithm, corrupted by an unknown noise distribution. |

#### 2. TRPCA Task (Sparse Denoising)
For the standard tensor denoising task, you need to load both the fully observed noisy data and the clean data for evaluation.
| penalty: Geman loss
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
