# 🤖 Imitation Learning for Path Following Using NVIDIA JetBot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Nano-green.svg)](https://developer.nvidia.com/embedded/jetson-nano)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An autonomous path-following system for the NVIDIA JetBot using behavioral cloning. The robot learns to navigate a track using only RGB camera input by imitating human demonstrations.

<p align="center">
  <img src="images/demo.gif" alt="JetBot Demo" width="600">
</p>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Training](#training)
  - [Deployment](#deployment)
- [Results](#results)
- [Report & Presentation](#report--presentation)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements an end-to-end autonomous driving system for the JetBot platform using behavioral cloning. Key highlights:

- **Algorithm**: Behavioral Cloning (BC) with DAgger for distribution shift correction
- **Model**: ResNet18 (pretrained on ImageNet) with dual-output head
- **Inputs**: Single RGB image (224×224×3)
- **Outputs**: Steering [-1, +1] and Speed factor [0, 1]
- **Performance**: 5+ minutes of continuous autonomous driving

## ✨ Features

- 📸 **Real-time inference** at 20 Hz on Jetson Nano
- 🎮 **Analog joystick control** for smooth data collection
- 🔄 **DAgger implementation** for iterative policy improvement
- 🚗 **Adaptive speed control** — slows in corners, accelerates on straights
- 📊 **Comprehensive training visualization** — loss curves, telemetry, saliency maps

## 🔧 Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Compute | NVIDIA Jetson Nano 4GB |
| Camera | IMX219 RGB (8MP) |
| Motors | 2× DC motors (differential drive) |
| Controller | Logitech F710 (analog) |
| Track | Black surface, white boundaries, red corner markers |

<p align="center">
  <img src="images/jetbot_full.jpg" alt="JetBot" width="300">
  <img src="images/track.jpg" alt="Track" width="300">
</p>

## 💻 Installation

### On PC (for training)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/jetbot-imitation-learning.git
cd jetbot-imitation-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### On JetBot (for data collection & deployment)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/jetbot-imitation-learning.git
cd jetbot-imitation-learning

# Install JetBot-specific dependencies
pip install -r requirements_jetbot.txt
```

### Dependencies

**PC (Training):**
```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

**JetBot (Deployment):**
```
torch  # JetPack version
torchvision
opencv-python
numpy
traitlets
jupyter
```

## 📁 Project Structure

```
jetbot-imitation-learning/
├── notebooks/
│   ├── jetbot_data_collection.ipynb    # Data collection on JetBot
│   ├── pc_training_notebook.ipynb      # Steering-only training
│   ├── pc_steering_speed_notebook.ipynb # Dual-output training
│   └── jetbot_deployment.ipynb         # Real-time inference
│
├── datasets/
│   ├── dataset_v1/                     # Initial dataset (8,314 images)
│   ├── dataset_dagger/                 # DAgger corrections (8,497 images)
│   └── dataset_steering_speed_v1/      # Final dataset (11,039 images)
│
├── models/
│   ├── steering_model_v1.pth           # Steering-only model
│   ├── steering_model_dagger.pth       # DAgger-enhanced model
│   └── steering_speed_model.pth        # Final dual-output model
│
├── images/                             # Figures for report/README
│
├── report/
│   ├── main.tex                        # LaTeX report
│   └── images/                         # Report figures
│
├── presentation/
│   └── slides.pdf                      # Presentation slides
│
├── requirements.txt
├── requirements_jetbot.txt
└── README.md
```

## 🚀 Usage

### Data Collection

1. Connect the Logitech F710 controller to the JetBot
2. Open `notebooks/jetbot_data_collection.ipynb`
3. Configure settings:
   ```python
   DATASET_DIR = 'dataset_steering_speed_v1'
   CROP_TOP = 0.20
   CROP_LEFT = 0.08
   CROP_RIGHT = 0.12
   ```
4. Hold **RB** to enable driving/recording
5. Use **Left Stick** for steering, **Right Trigger** for speed

### Training

1. Transfer dataset from JetBot to PC
2. Open `notebooks/pc_steering_speed_notebook.ipynb`
3. Configure paths:
   ```python
   DATASET_DIR = 'dataset_steering_speed_v1'
   MODEL_SAVE_PATH = 'steering_speed_model.pth'
   ```
4. Run all cells to train
5. Training takes ~35 minutes on a consumer GPU

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Weight Decay | 1e-5 |
| Early Stopping | Patience = 7 |

### Deployment

1. Transfer trained model to JetBot
2. Open `notebooks/jetbot_deployment.ipynb`
3. Load model and run inference loop:
   ```python
   model = load_model('steering_speed_model.pth')
   # Runs at ~20 Hz
   ```

### DAgger (Optional - for improving corners)

1. Deploy initial model
2. When robot fails, take manual control while continuing to record
3. Merge new data with original dataset
4. Retrain

## 📊 Results

### Training Metrics

| Model | Dataset | MSE (Steering) | MAE (Steering) | MSE (Speed) | MAE (Speed) |
|-------|---------|----------------|----------------|-------------|-------------|
| Steering-only | 8,314 | 0.0265 | 0.0917 | — | — |
| + DAgger | 8,497 | 0.0265 | 0.0892 | — | — |
| + Speed (Final) | 11,039 | 0.0318 | 0.1106 | 0.0080 | 0.0293 |

### Training Curves

<p align="center">
  <img src="images/loss_LR.png" alt="Training Curves" width="700">
</p>

### Real-World Performance

- ✅ **5+ minutes** continuous autonomous driving
- ✅ **Multiple laps** without intervention
- ✅ **Adaptive speed** — slows for corners, accelerates on straights
- ✅ **20 Hz** inference on Jetson Nano
- ✅ **First-attempt success** for dual-output model

### Demo Videos

| Description | Link |
|-------------|------|
| Final model (steering + speed) | [Watch](https://youtube.com/...) |
| Before DAgger (corner failures) | [Watch](https://youtube.com/...) |
| After DAgger (corners fixed) | [Watch](https://youtube.com/...) |

## 📄 Report & Presentation

- **Full Report**: [report/main.pdf](report/main.pdf)
- **Presentation Slides**: [presentation/slides.pdf](presentation/slides.pdf)

## 👥 Authors

| Name | Student ID |
|------|------------|
| Adel Saidani | U6104239 |
| Enis Hedri | — |
| Mahra Alhosani | U1100303 |

**Supervisors:**
- Dr. Narcis Palomeras Rovira
- Marta Real Vial

**Institution:** University of Girona — Master in Intelligent Robotic Systems (MIRS)

## 🙏 Acknowledgments

- [NVIDIA JetBot](https://github.com/NVIDIA-AI-IOT/jetbot) — Open-source robot platform
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Ross et al., 2011](https://arxiv.org/abs/1011.0686) — DAgger algorithm

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>University of Girona — Master in Intelligent Robotic Systems (MIRS) — 2025</b>
</p>
