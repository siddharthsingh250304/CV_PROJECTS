# **Deep Learning Experimentations for Computer Vision and NLP**

This repository contains an extensive collection of experiments leveraging deep learning models for **computer vision** and **natural language processing (NLP)** tasks. The work presented here is the culmination of assignments and projects completed as part of an advanced **Computer Vision** course. The experiments span across **image classification**, **object detection**, **semantic segmentation**, and **vision transformers** (ViT), along with **CLIP** for zero-shot learning. The project is designed to explore model architectures, performance evaluation techniques, and visualizations, providing comprehensive insights into cutting-edge deep learning research and applications.

---

## **Table of Contents**

1. **[Classification with ResNet](#classification-with-resnet)**
2. **[Object Detection](#object-detection)**
3. **[Semantic Segmentation](#semantic-segmentation)**
4. **[Vision Transformers (ViT) & CLIP](#vision-transformers-vit--clip)**
5. **[Conclusion](#conclusion)**

---

## **1. Classification with ResNet**

### **Objective**  
This section focuses on training the **ResNet18** model for image classification on a custom 36x36 dataset. The project compares **training from scratch** and **fine-tuning with ImageNet pre-trained weights** to assess performance.

### **Experiments**  
- **Training from Scratch** vs. **ImageNet Pretrained ResNet18**
- **Resizing Strategies**: Evaluated different resizing techniques (224x224 vs. 36x36).
- **Model Architecture Modifications**: Experimented with different architectural variations.
- **Evaluation Metrics**: Loss, accuracy, F1 score, confusion matrices.
- **Logging**: All experiments were logged using **Weights & Biases**.

**File**: `Q1_ResNet_Classification.ipynb`

---

## **2. Object Detection**

### **Objective**  
This section focuses on enhancing **Faster R-CNN** to predict **oriented bounding boxes (OBBs)**, alongside detecting and counting fruits and human body parts using instance segmentation masks.

### **Key Experiments**  
1. **Oriented Bounding Box Prediction**  
   - Extended Faster R-CNN to predict oriented bounding boxes.
   - Modified the ROIHeads to predict angles along with bounding box coordinates.
   - Visualizations of RPN objectness maps and proposal evolution.
   - Evaluation with extended **mAP**, **precision**, **recall**, and **angle loss**.

   **Directory**: `Oriented_BBox_Prediction/`

2. **Fruit Detection and Counting**  
   - Leveraged **instance segmentation masks** converted to bounding boxes.
   - Implemented on-the-fly augmentation techniques like rotation, flipping, and color jitter.
   - Evaluated model performance using **mAP** and **precision-recall**.

   **File**: `Fruit_detection_and_counting.ipynb`

3. **Human Parts Detection**  
   - Converted human part segmentation masks into bounding box annotations.
   - Performed analysis and visualization of part-specific distribution, scale, and aspect ratios.
   - Evaluated model performance with part-wise **precision** and **recall**.

   **File**: `Human_detection.ipynb`

---

## **3. Semantic Segmentation**

### **Objective**  
This section involves multiple experiments on semantic segmentation, implementing **Fully Convolutional Networks (FCNs)** and **U-Net variants** to segment 13 distinct classes in a custom dataset.

### **Key Experiments**  
1. **FCN Models**  
   - Implemented **FCN-32s**, **FCN-16s**, and **FCN-8s** using **VGG16/19** backbones.
   - Evaluated **frozen encoder** vs **finetuned encoder** configurations.
   - Performed evaluation using **Mean IoU** and visualizations of predicted vs ground truth masks.

   **File**: `FCN_Segmentation.ipynb`

2. **U-Net Based Models**  
   - Implemented and compared multiple variants of **U-Net**, including **Vanilla U-Net**, **Residual U-Net**, **No-Skip U-Net**, and **Gated Attention U-Net**.
   - Evaluated performance across different segmentation tasks with visual comparisons.
   - Utilized attention mechanisms to improve segmentation accuracy and visual clarity.

   **File**: `U_Net_Segmentation.ipynb`

---

## **4. Vision Transformers (ViT) & CLIP**

### **Objective**  
This section explores **Vision Transformers (ViT)** and their extension **Differential ViT (Diff-ViT)**, along with **CLIP** for zero-shot image classification, across multiple datasets.

### **Key Experiments**  
1. **Vision Transformer (ViT)**  
   - Implemented **ViT** on **CIFAR-10** with **multi-head attention**, **patch embedding**, and various **positional embedding strategies**.
   - Experimented with patch sizes, data augmentations, and hyperparameter tuning to achieve >80% accuracy within 50 epochs.
   - Visualized **attention maps**, **DINO attention**, and **attention rollout**.

   **File**: `ViT_Classification.ipynb`

2. **Differential Vision Transformer (Diff-ViT)**  
   - Extended **ViT** with a **Differential Attention Mechanism** to improve information amplification and noise cancellation.
   - Compared performance against the original ViT, evaluating various configurations and hyperparameters.

   **File**: `Diff_ViT_Classification.ipynb`

3. **CLIP (Contrastive Language-Image Pretraining)**  
   - Applied **CLIP** for **zero-shot image classification** using a **pretrained ResNet-50** model.
   - Evaluated the impact of **FP16** vs **FP32** precision formats on performance, analyzing **memory usage** and **speed**.
   - Compared **CLIP**'s performance against **ImageNet-pretrained models** in a variety of image categories.

   **File**: `CLIP_ZeroShot_Classification.ipynb`

---

## **5. Conclusion**


This repository represents the cumulative efforts of the work done by me for the **Computer Vision** course.

---
