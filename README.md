# 🤟 Benchmark & Skeleton Sequence Modeling for Isolated Sign Language Recognition on WLASL

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Machine Learning & Deep Learning Final Research Project**  
> *Department of Computer Science & Engineering, K. N. Toosi University of Technology*

---

## 📌 Executive Summary & Quick Links

Isolated Sign Language Recognition (ISLR) is critical for bridging communication divides between deaf and hearing communities[cite: 6]. However, training high-capacity 3D RGB Vision Transformers on unconstrained web-scraped video datasets leads to catastrophic pixel-level overfitting and extreme computational latency[cite: 6]. 

This project delivers a comprehensive empirical study on the **World-Level American Sign Language (WLASL)** benchmark[cite: 6]:
1. **Vision Baseline:** Re-implementing a **Video Swin Transformer (Swin3D-T)** with YOLOv8 subject cropping on full WLASL-2000[cite: 6].
2. **Landmark Sequence Pivot:** Engineering a lightweight pipeline combining **MediaPipe Holistic tracking**, bounded temporal linear gap-filling, nose-anchored spatial normalization, and horizontal sequence mirroring[cite: 6].
3. **Sequence Architecture:** Introducing **RobustConvBiLSTM**—a 1D-Temporal Convolutional Bidirectional LSTM with dual temporal pooling[cite: 6].

### 🔗 Project Deliverables & Media
* 🎥 **Product Pitch Video (YouTube):** [Watch on YouTube](https://youtube.com/) *(Replace with actual link)*
* 🎬 **Product Pitch Video (Aparat):** [Watch on Aparat](https://aparat.com/) *(Replace with actual link)*
* 📂 **Full Physical Archives (Google Drive):** [Access Google Drive Folder](https://drive.google.com/) *(Replace with actual link)*
* 📄 **Technical Report (PDF):** [Download Technical PDF](./docs/ML_RV_Final_Project.pdf)[cite: 6]
* 📊 **Presentation Slides (PDF):** [Download Presentation Slides](./docs/Presentation_Slides.pdf)

---

## 👥 Authors & Group Members

* **Mobina Yousefi Moghadam**[cite: 6]
* **Iman Bidi**[cite: 6]
* **Yazdan Bayat**[cite: 6]

---

## 🔬 System Pipeline Architecture
