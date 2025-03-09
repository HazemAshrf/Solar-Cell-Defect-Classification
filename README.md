# Solar Cell Defect Classification: Multi-Model Sweep with PyTorch

This repository implements a **multi-model deep learning pipeline** using PyTorch to classify defects in solar cells. The project compares **ResNet-like, EfficientNetB3, EfficientNetV2M, and ConvNeXtBase** models using **advanced augmentation, weighted sampling, and a hyperparameter sweep with Weights & Biases (WandB)**.

## Dataset and Task

- **Images**: Electroluminescence images of solar cells (`images.zip`), converted from grayscale to RGB.  
- **Labels**: Provided in `data.csv` with two binary labels:  
  - `crack`: 1 if the cell has visible cracks.  
  - `inactive`: 1 if part of the cell is inactive.  
- **Multi-label classification**: Each image may have **both defects simultaneously**.

## Implemented Models

1. **ResNet-like Model** – Custom residual blocks tailored for this task.  
2. **EfficientNetB3** – Pretrained CNN with modified classifier.  
3. **EfficientNetV2M** – Optimized version of EfficientNet for improved efficiency.  
4. **ConvNeXtBase** – State-of-the-art vision model with hierarchical feature extraction.  

## Training Pipeline & Techniques

- **WeightedRandomSampler**: Handles class imbalance.  
- **MixUp & CutMix**: Applied via a custom `collate_fn` for richer augmentation.  
- **AdamW Optimizer**: Uses differential learning rates for feature extractor vs. classifier.  
- **ReduceLROnPlateau Scheduler**: Dynamically adjusts learning rate based on loss.  
- **Early Stopping**: Prevents overfitting by monitoring validation loss.  
- **WandB Hyperparameter Sweep**: Automates model selection for best performance.  

## Results & Experiment Tracking

- **Performance Metrics**: Logs **F1-score, accuracy, and validation loss** per epoch.
- **WandB Integration**: Tracks experiment results and manages model checkpoints.
- **ONNX Model Export**: Saves trained models for deployment.

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is developed as part of a deep learning challenge, leveraging PyTorch and modern CNN architectures for defect classification.

