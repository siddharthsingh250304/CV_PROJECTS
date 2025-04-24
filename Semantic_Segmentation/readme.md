# 🧠 Semantic Segmentation Projects

## 📁 Dataset Overview
- Semantic segmentation dataset with **13 classes**:
  - Unlabeled, Building, Fence, Other, Pedestrian, Pole, Roadline, Road, Sidewalk, Vegetation, Car, Wall, Traffic sign.
- Image sizes:
  - **224×224** (FCN)
  - **256×256** (U-Net)
- Masks are single-channel images with pixel values from 0–12 indicating class IDs.

---

## 🚀 Fully Convolutional Networks (FCNs)

### 📊 Dataset Visualization
- Developed a tool to visualize binary masks for each class.
- Enabled inspection of per-class pixel distribution across the dataset.

### 🏗️ Model Variants Implemented
- **FCN-32s**
- **FCN-16s**
- **FCN-8s**
- Encoder: **VGG16/VGG19 pretrained on ImageNet**

### 🔧 Training Regimes
- **Frozen Encoder**: Trained with the backbone frozen.
- **Finetuned Encoder**: Enabled backpropagation through the full network.

### 📈 Evaluation & Visualization
- Metrics: **Mean IoU (mIoU)**, Loss
- Used `torchmetrics.MeanIoU` for accuracy tracking.
- Visualized predictions vs ground truth for qualitative evaluation.

### 🧪 Key Comparisons
- Compared segmentation quality across FCN variants.
- Analyzed the impact of freezing vs finetuning the encoder.

---

## 🧠 U-Net Based Models

### 🔨 Vanilla U-Net
- Implemented a 4-level U-Net with:
  - Encoder, bottleneck, decoder
  - Two convolutional layers per resolution level
  - Transposed convolutions for upsampling
  - Skip connections for feature reuse

### ❌ U-Net without Skip Connections
- Removed skip connections to evaluate their role.
- Compared mIoU and visual outputs with Vanilla U-Net.

### 🔁 Residual U-Net
- Replaced standard conv blocks with residual blocks.
- Used 1×1 convolutions to match dimensions when needed.

### 🎯 Gated Attention U-Net
- Integrated additive attention gates in skip connections.
- Followed original paper with coefficient α = 1.
- Analyzed performance improvements and visual clarity.

### 📈 Results & Visualizations
- Tracked **loss and mIoU** over training.
- Visualized predicted masks vs ground truth for each U-Net variant.
- Compared Vanilla, No-Skip, Residual, and Attention U-Nets on performance and segmentation quality.

---

