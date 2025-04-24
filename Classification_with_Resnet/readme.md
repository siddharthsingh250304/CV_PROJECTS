# 🧠 ResNet Analysis & Style Transfer on Custom Dataset

This project explores convolutional neural networks, interpretability techniques, and neural style transfer using PyTorch. It is divided into three main parts:

## 📌 Contents

### 🔹 Q1: ResNet18 Adaptation and Training
File: `Q1_ResNet_Classification.ipynb`

- Trains ResNet18 from scratch and with ImageNet pretrained weights on a 36x36 custom dataset.
- Compares baseline training, resizing strategies (224x224), and architecture modifications.
- Logs all experiments using [Weights & Biases](https://wandb.ai/) for loss and accuracy plots.
- Discusses the impact of different architectural changes and dataset preprocessing on performance.
- Computes F1 score and confusion matrices for final evaluations.

### 🔹 Q2: Network Visualization
File: `Q2_ResNet_Visualisation.ipynb`

- Uses saliency maps to visualize model attention per class.
- Masks input images with black or noise to observe model behavior.
- Performs adversarial attacks by adding noise or optimizing image pixels.
- Visualizes original, adversarial, and perturbed outputs with interpretations.

### 🔹 Q3: Neural Style Transfer
File: `Q3_Style_Transfer.ipynb`

- Implements style transfer based on the method from [Gatys et al., 2015](https://arxiv.org/abs/1508.06576).
- Calculates content and style loss using VGG19 layers.
- Optimizes using L-BFGS and compares results with Adam optimizer.
- Transfers styles to various images, including a self-portrait, and experiments with multiple style-content weight configurations.

---

## 🧪 Dependencies
- Python 3.8+
- PyTorch >= 1.12
- torchvision
- matplotlib
- numpy
- wandb

Install via:
```bash
pip install torch torchvision matplotlib numpy wandb

