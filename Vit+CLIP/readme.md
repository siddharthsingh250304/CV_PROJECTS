## Vision Transformer (ViT)

### Overview
This section involves the implementation and training of the Vision Transformer (ViT) model from scratch on the CIFAR-10 dataset, based on the paper [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929).

### Key Components:
1. **Transformer Encoder**: A multi-layer transformer encoder for learning complex image representations.
2. **MLP Head**: The final classification head composed of a multi-layer perceptron.
3. **Patch Embedding**: Input image is divided into patches, which are linearly embedded.
4. **Position Embedding**: Adds positional information to patch embeddings (varied positional embedding strategies explored).
5. **[CLS] Token**: A special token prepended to the input sequence to aggregate global information.

### Tasks:
- **Scaled Dot Product Attention and Multi-Head Attention**: Implemented as per ViT paper.
- **Patch Sizes**: Experimentation with patch sizes of 2, 4 (default), and 8.
- **Hyperparameter Tuning**: Exploration of embedding dimensions, transformer layers, MLP hidden dimensions, etc., to achieve >80% test accuracy in 50 epochs.
- **Data Augmentations**: Examining different data augmentation techniques to improve performance.
- **Positional Embedding Types**: No positional embedding, 1D learned, 2D learned, and sinusoidal.

### Visualizations:
- **Attention Maps**: Visualized attention maps from the [CLS] token to patch tokens for different heads and layers.
- **DINO Attention**: Used DINO for attention map visualizations of pre-trained ViT.
- **Attention Rollout**: Implemented as per the ViT paper to compute attention flow.

### Experiments and Results:
- Hyperparameter exploration, data augmentation, and positional embedding variations were conducted to optimize performance. Detailed loss curves and test accuracy are documented.

## Differential Vision Transformer (Diff-ViT)

### Overview
Diff-ViT extends the Vision Transformer by incorporating Differential Attention Mechanism, introduced in the [Differential Transformer Paper](https://arxiv.org/abs/2410.05258). This mechanism amplifies important information while canceling out noise, similar to noise-canceling headphones.

### Tasks:
1. **Differential Attention Mechanism**: Implemented multi-head differential attention as described in the paper.
2. **Integration**: Replaced standard multi-head attention with the differential attention in ViT.
3. **Training**: Diff-ViT model trained on CIFAR-10, with experiments on patch size variations and positional embeddings.

### Experiments and Results:
- Similar experiments to ViT were conducted for Diff-ViT, with comparisons in terms of hyperparameters, data augmentation, and positional embeddings.
- Performance comparison between standard ViT and Diff-ViT models was made, with results included in the project.

### Visualizations:
- Attention map visualizations and attention rollout as per ViT, but using the Diff-ViT architecture.

## CLIP

### Overview
CLIP (Contrastive Language-Image Pretraining) is used for zero-shot image classification, leveraging both vision and text encoders.

### Tasks:
1. **Inference with CLIP**: Inference was conducted using the pre-trained ResNet-50 model, both with ImageNet pretraining and OpenAI’s CLIP.
2. **Zero-Shot Classification**: CLIP was used for zero-shot classification on ImageNet.
3. **FP16 Conversion**: CLIP's image encoder was converted to FP16 to analyze the speed and memory usage difference from FP32.

### Experiments:
- Comparison between CLIP and ImageNet pretraining for different image categories.
- Evaluated CLIP’s performance with a variety of images, including cases where CLIP outperforms ImageNet pretraining and vice versa.
- Performance was tested using both FP16 and FP32 precision formats, analyzing time and memory usage differences.

## Results
The best configurations, including model hyperparameters, training strategies, and augmentation techniques, are documented. The most successful ViT and Diff-ViT configurations achieved over 80% accuracy on CIFAR-10 within 50 epochs.

## Conclusion
The repository provides implementations for Vision Transformer and its differential variant, with a comprehensive set of experiments on CIFAR-10. Additionally, CLIP was explored for zero-shot image classification. Visualizations, including attention maps and rollout analysis, help in understanding the models’ decision-making processes.
