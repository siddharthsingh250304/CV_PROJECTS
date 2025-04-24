## 📁 Oriented Bounding Box Prediction (`Oriented_BBox_Prediction/`)

### 📝 Objective
Extend the standard **Faster R-CNN** architecture to predict **oriented bounding boxes** (OBBs) rather than axis-aligned ones.

### ✅ Key Features
- **Architecture Changes**:
  - Modified `ROIHeads` to include orientation (angle) prediction.
  - Added both **angle regression** and **multi-bin classification** methods.
- **Visualizations**:
  - RPN objectness maps and proposal evolution as animations.
  - Anchor assignment (positive vs. negative) visualization.
  - Comparison of RPN vs final ROI head outputs.
- **Metrics & Evaluation**:
  - Extended mAP to handle oriented boxes.
  - Logged Precision, Recall, and Angle Loss.
  - Hyperparameter tuning for angle loss weights and bin sizes.

---

## 📓 Fruit Detection and Counting (`Fruit_detection_and_counting.ipynb`)

### 📝 Objective
Detect and count fruits using instance segmentation masks converted into bounding boxes.

### ✅ Key Features
- **Preprocessing**:
  - `masks_to_boxes()` function to extract bounding boxes from segmentation masks.
  - Handled overlapping instances and edge cases.
- **Dataset & Augmentation**:
  - Custom `FruitDetectionDataset` with:
    - On-the-fly conversion of masks to boxes.
    - Augmentations: rotation, flip, color jitter.
- **Model**:
  - **Faster R-CNN** with a **ResNet-34** backbone (pretrained).
  - Custom anchor box configurations for fruit detection.
- **Evaluation**:
  - Visualized detections, segmentation masks, and comparisons.
  - mAP, Precision-Recall, and qualitative analysis under occlusion, lighting, and scale variations.

---

## 📓 Human Parts Detection (`Human_detection.ipynb`)

### 📝 Objective
Detect different human body parts from part segmentation masks.

### ✅ Key Features
- **Data Processing**:
  - Converted part segmentation masks to bounding box annotations.
  - Created data structures for part-wise labeling and dataset splits.
- **EDA**:
  - Analyzed part distribution, scale, and aspect ratios.
  - Visualized part annotations and patterns in the dataset.
- **Model Architecture**:
  - **Faster R-CNN** base.
  - Adapted anchors or priors specifically for body parts.
- **Training & Analysis**:
  - Trained and tuned the model with augmentation.
  - Evaluated mAP, part-specific precision and recall.
  - Investigated failure cases (e.g. occlusion, viewpoint variation).
