# 🤟 Benchmark & Skeleton Sequence Modeling for Isolated Sign Language Recognition on WLASL

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Machine Learning & Deep Learning Final Research Project**  
> *Department of Computer Science & Engineering, K. N. Toosi University of Technology*

---

## 📄 Table of Contents
1. [Introduction](#1-introduction)
2. [Authors & Contributors](#2-authors--contributors)
3. [Dataset Exploration & Characteristics](#3-dataset-exploration--characteristics)
4. [Methodology & Pipeline Progression](#4-methodology--pipeline-progression)
   - [Phase 1: Vision Baseline (YOLOv8 + Swin3D-T)](#phase-1-vision-baseline-yolov8--swin3d-t)
   - [Phase 2: Landmark Sequence Pivot (MediaPipe)](#phase-2-landmark-sequence-pivot-mediapipe)
   - [Phase 3: Preprocessing & Data Quality Control](#phase-3-preprocessing--data-quality-control)
   - [Phase 4: Sequence Model (RobustConvBiLSTM)](#phase-4-sequence-model-robustconvbilstm)
5. [Empirical Results & Comparative Analysis](#5-empirical-results--comparative-analysis)
   - [Performance Comparison on WLASL-100](#performance-comparison-on-wlasl-100)
   - [Full WLASL-2000 Baseline vs. Literature](#full-wlasl-2000-baseline-vs-literature)
6. [Repository Structure](#6-repository-structure)
7. [Installation & Usage](#7-installation--usage)
8. [License](#8-license)

---

## 1. Introduction

Isolated Sign Language Recognition (ISLR) is essential for bridging communication divides between deaf and hearing communities. However, training high-capacity 3D RGB Vision Transformers on raw, unconstrained web videos introduces significant hurdles:
* **Pixel Overfitting:** RGB networks memorize background furniture, room lighting, and signer clothing rather than motion trajectories.
* **High Computational Latency:** Processing dense $3D$ video tensors ($32 \times 224 \times 224$) introduces massive VRAM and compute overhead.
* **Extreme Data Scarcity:** Real-world ASL datasets suffer from long-tailed class distribution with very few video instances per word.

This project presents a systematic empirical progression: we first re-implement a **Video Swin Transformer (Swin3D-T)** with **YOLOv8 signer cropping** on the full 2,000-word dataset, analyze its fundamental limitations, and then introduce a lightweight **Skeletal Landmark Sequence Pipeline (RobustConvBiLSTM)** that strips background clutter and achieves state-of-the-art accuracy at sub-second training speeds.

---

## 2. Authors & Contributors

* **Mobina Yousefi Moghadam**
* **Iman Bidi**
* **Yazdan Bayat**

---

## 3. Dataset Exploration & Characteristics

This study utilizes the [WLASL (World Level American Sign Language) Video Dataset by risangbaskoro on Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed), containing video instances paired with `WLASL_v0.3.json` metadata.

### Dataset Statistics
* **Total Unique Glosses:** 2,000 unique sign words.
* **Total Video Instances:** 21,083 annotated video clips.
* **Average Instances per Gloss:** $\approx 10.54$ samples/word.
* **Class Sample Imbalance:** Max instance class = 16 samples (e.g., 'book') down to 3–5 samples for tail classes.
* **Physical Video Dynamics:** Frame rates span 15–60 FPS (Mode: 25.0/29.97 FPS), durations span 0.45s–6.13s (Mean: 1.58s), with unconstrained resolutions ($1280 \times 720$, $640 \times 480$, $320 \times 240$).

---

## 4. Methodology & Pipeline Progression
[Raw RGB Video Stream]
│
▼
[Stage 1: YOLOv8 Signer Isolation] ──> Crops upper body & normalizes subject scale (224x224)
│
▼
[Stage 2: MediaPipe Extraction]   ──> Extracts 75 3D joints (D = 258 parameters/frame)
│
▼
[Stage 3: Advanced Preprocessing]  ──> (1) Bounded Temporal Linear Interpolation
(2) Nose-Centering & Shoulder Width Scaling
(3) Left/Right Hand Sequence Mirroring
│
▼
[Stage 4: RobustConvBiLSTM Model]  ──> 1D-Conv Smoothing + 2-Layer BiLSTM + Dual Pooling
│
▼
[Predicted Sign Gloss Classification]
### Phase 1: Vision Baseline (YOLOv8 + Swin3D-T)
Raw videos in WLASL vary in camera distance and framing. To prevent self-attention layers from wasting compute on background artifacts, we implement a uniform sampling (T=32 frames) and detection pipeline using **YOLOv8**. The upper-body signer region (`cls=0`) is cropped, clamped, and bilinearly resized to 224 x 224.

![YOLO Cropped Frames Sequence](./assets/yolo_cropped_sequence.png)  
*Figure 1: Sample extracted keyframe sequence (Video ID: 20979) showing signer isolation and background filtering across 32 keyframes.*

The cropped frames feed into a **Video Swin Transformer (Swin3D-T)** initialized with Kinetics-400 pre-trained weights. The final classification head is reprojected to N=2,000 outputs using Xavier Normal initialization and optimized via Automatic Mixed Precision (AMP) and Differential Learning Rates (eta_backbone = 3e-5, eta_head = 3e-4).

### Phase 2: Landmark Sequence Pivot (MediaPipe)
To eliminate background pixel noise entirely, we transition from RGB volumes to sparse skeletal trajectories. Using **MediaPipe Pose Heavy** and **MediaPipe Hand Landmarkers**, each video frame is reduced to a 1D feature array of length D=258:
* **Upper-Body Pose (P_pose):** 33 joints x 4 values (x, y, z, v) = 132 parameters (Indices 0–131).
* **Left Hand (H_left):** 21 joints x 3 values (x, y, z) = 63 parameters (Indices 132–194).
* **Right Hand (H_right):** 21 joints x 3 values (x, y, z) = 63 parameters (Indices 195–257).

### Phase 3: Preprocessing & Data Quality Control
Raw keypoint tracking suffers from occlusions and camera shift. We apply a three-stage geometric normalization pipeline:
1. **Quality Audit & Pruning:** Purges clips with sequence length T < 15 frames or active hand presence P_hand < 30%.
2. **Temporal Linear Interpolation:** Fills internal tracking dropouts (0.0 coordinate spikes) using bounded linear interpolation.
3. **Two-Stage Spatial Normalization:** Re-anchors origin (0,0,0) to the Nose landmark and scales all joints by Euclidean shoulder-width distance d_shoulder.
4. **Horizontal Mirroring:** Inverts X-coordinates and swaps left/right hand slots to eliminate handedness bias.

### Phase 4: Sequence Model (RobustConvBiLSTM)
The preprocessed 516-dimensional array (positions + temporal velocity derivatives) feeds into **RobustConvBiLSTM**:
1. **1D Temporal Convolutional Prefix (K=3, P=1, H_conv=128):** Smoothes frame-to-frame velocity representations.
2. **2-Layer Bidirectional LSTM (H_lstm=128):** Contextualizes gesture trajectories forwards and backwards.
3. **Dual Temporal Pooling:** Concatenates temporal mean-pooling and max-pooling into a unified 512-dim vector.
4. **Classification Head:** Linear -> BatchNorm -> ReLU -> Dropout(0.4) -> Linear projection.

---

## 5. Empirical Results & Comparative Analysis

### Performance Comparison on WLASL-100
Our preprocessed **RobustConvBiLSTM** landmark network achieves a peak **Top-1 accuracy of 43.56%** and **Top-5 accuracy of 72.28%**. This represents a massive **+19.62% Top-1 increase over the Video Swin Transformer** while executing **75x faster per epoch**.

| Model Architecture | Input Modality | Top-1 Acc. (%) | Top-5 Acc. (%) | Epoch Execution Time | Deployability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pose-TGCN (Baseline)** | Raw MediaPipe Vectors | 12.87% | 33.66% | ~5.0s | Edge / Real-time |
| **Swin3D-T (Vision Baseline)** | YOLO-Cropped RGB | 23.94% | 59.81% | ~225.0s | High GPU Overhead |
| **RobustConvBiLSTM (Ours)** | **Processed Keypoints** | **43.56%** | **72.28%** | **~3.0s** | **Real-Time Edge** |

### Full WLASL-2000 Baseline vs. Literature
Benchmark comparison of our vision baseline on the full 2,000-class WLASL dataset against published literature:

| Model Architecture | Input Modality | Classes (N) | Test Top-1 (%) | Test Top-5 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Chance** | Uniform Distribution | 2,000 | 0.05% | 0.25% |
| **VGG-16 + GRU** (Li et al., 2020) | 2D RGB + Recurrent | 2,000 | 8.44% | 23.58% |
| **Pose-GRU** (Li et al., 2020) | 2D OpenPose Keypoints | 2,000 | 22.54% | 49.81% |
| **Pose-TGCN** (Li et al., 2020) | Graph Convolutional | 2,000 | 23.65% | 51.75% |
| **I3D (RGB Fine-tuned)** (Li et al., 2020) | 3D RGB Frame Stacks | 2,000 | 32.48% | 57.31% |
| **Swin3D-T (Ours)** | **YOLO-Cropped RGB** | **2,000** | **23.94%** | **59.81%** |

### Key Takeaways
1. **+2.50% Top-5 Boost over Paper Baseline:** Swin3D-T with YOLO cropping outperforms the best 3D ConvNet from the original paper (I3D: 57.31% vs. Ours: 59.81%).
2. **Elimination of Pixel Overfitting:** Stripping pixels via MediaPipe prevents models from memorizing background furniture, improving Top-1 accuracy to 43.56%.
3. **Sub-Second Efficiency:** Keypoint sequence training executes in ~3.0s per epoch vs. 225.0s for 3D Video Swin.

---

## 6. Repository Structure

```text
.
├── assets/                       # Visual assets, figures, and diagrams
│   └── yolo_cropped_sequence.png
├── configs/                      # Experiment YAML configuration files
│   └── base.yaml
├── docs/                         # Written report and presentation slides
│   ├── ML_RV_Final_Project.pdf
│   └── Presentation_Slides.pdf
├── notebooks/                    # Colab / Jupyter notebooks
│   └── WLASL_Pipeline.ipynb
├── src/                          # Modular Python source code
│   ├── dataset.py                # PyTorch DataLoaders & GroupShuffleSplit
│   ├── evaluator.py              # Metrics evaluation & confusion matrix generator
│   ├── extract_keypoints.py      # MediaPipe batch extraction script
│   ├── model.py                  # Swin3D-T & RobustConvBiLSTM architectures
│   ├── preprocess.py             # Interpolation, nose-centering & mirroring logic
│   ├── prune_dataset.py          # Automated dataset health audit
│   └── trainer.py                # AMP training routines & differential LRs
├── .gitignore                    # Git rules for ignoring large files
├── LICENSE                       # MIT License
├── README.md                     # Main repository documentation
├── requirements.txt              # Dependency manifest
└── train.py                      # Central execution entry point
