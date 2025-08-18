# LLFlow: Infrared Low-Light Image Enhancement with Conditional Normalizing Flows

A implementation of a conditional normalizing flow model for infrared low-light image enhancement. This project uses invertible neural networks to learn the transformation between infrared low-light and normally exposed images, enabling high-quality enhancement of dark images.

## Model Architecture

The model implements a conditional normalizing flow with the following key components:

- **Conditional Prior**: Uses μ(x) = prior_mu(condition_encoder(x)) for meaningful latent space sampling
- **Invertible Coupling Layers**: Affine transformations conditioned on input infrared low-light images
- **Multi-Loss Training**: Combines negative log-likelihood, L1 reconstruction, and color alignment losses
- **Stable Training**: Bounded coupling scales and proper log-determinant accumulation

## Dataset

The model is trained on paired infrared low-light and normal-light image datasets:

### Training Dataset
- **Location**: `train-20241116T084056Z-001/train-20241116T084056Z-001/train/`
- **Structure**: 
  - `low/`: 236 infrared low-light input images
  - `high/`: 236 corresponding well-exposed target images
- **Format**: PNG images, various resolutions
- **Preprocessing**: Resized to 128x128 during training for computational efficiency

### Evaluation Dataset
- **Location**: `eval-20241116T084057Z-001/eval-20241116T084057Z-001/eval/`
- **Structure**:
  - `low/`: 87 infrared low-light test images
  - `high/`: 87 corresponding ground truth images
- **Usage**: Model evaluation and performance benchmarking

The dataset contains diverse scenes including indoor/outdoor environments, different lighting conditions, and various object types, providing comprehensive coverage for robust model training.

## Key Features

- **Conditional Prior Sampling**: Eliminates gray output collapse through learned conditional priors
- **Invertible Architecture**: Ensures information preservation and stable training
- **Color Alignment Loss**: Maintains color distribution consistency using differentiable histograms
- **Comprehensive Evaluation**: PSNR, SSIM metrics with detailed result logging
- **Visualization Tools**: Automatic triptych generation (input | enhanced | target)

## Performance Results

### Model Performance Metrics
- **PSNR**: 18.07 dB
- **SSIM**: 0.7614
- **Training Time**: 30 epochs on CPU
- **Model Size**: ~29MB checkpoint

### Architectural Improvements
- **Conditional Prior μ(x)**: Prevents gray output collapse
- **Log-Determinant Accumulation**: Proper invertible flow training
- **Bounded Coupling Scales**: Enhanced training stability
- **Multi-Component Loss**: Balanced enhancement quality

## Installation

### Requirements
```bash
pip install torch torchvision numpy pillow tqdm scikit-image tensorboard
```

### Dependencies
- Python 3.7+
- PyTorch 2.0+
- NumPy
- PIL (Pillow)
- scikit-image
- TensorBoard
- tqdm

## Usage

### Training

```bash
python model_script.py --mode train --config cpu_train_config.json
```

### Evaluation

```bash
python model_script.py --mode evaluate --checkpoint checkpoints/best_model.pth --config cpu_train_config.json
```

### Single Image Enhancement

```bash
python model_script.py --mode enhance --checkpoint checkpoints/best_model.pth --input path/to/low_light.jpg --output path/to/enhanced.jpg --config cpu_train_config.json
```

### Demo Mode (Quick Test)

```bash
python model_script.py --mode demo --config cpu_train_config.json
```

## Configuration

The training configuration is managed through JSON files. Key parameters in `cpu_train_config.json`:

```json
{
  "device": "cpu",
  "use_amp": false,
  "num_workers": 0,
  "pin_memory": false,
  "batch_size": 1,
  "image_size": [128, 128],
  "flow_channels": 96,
  "num_flow_blocks": 2,
  "learning_rate": 0.0002,
  "val_freq": 1,
  "save_freq": 1,
  "num_epochs": 30
}
```

## Model Components

### ConditionalFlow
- **Condition Encoder**: Extracts features from infrared low-light inputs
- **Prior Network**: Learns conditional prior μ(x) for meaningful sampling
- **Flow Layers**: Invertible transformations for image enhancement
- **Projection Layers**: Input/output mappings with sigmoid activation

### Loss Functions
1. **Negative Log-Likelihood**: Flow-based reconstruction loss under N(μ(x), I)
2. **L1 Loss**: Pixel-wise reconstruction accuracy
3. **Color Alignment Loss**: Wasserstein distance between color histograms

### Training Features
- **Automatic Mixed Precision**: Optional AMP support for faster training
- **Gradient Clipping**: Prevents gradient explosion
- **Cosine Learning Rate Scheduling**: Smooth learning rate decay
- **Best Model Checkpointing**: Automatic saving based on validation PSNR

## Results Structure

```
results/
├── eval_YYYYMMDD_HHMMSS/
│   ├── images/
│   │   ├── *.png (enhanced images)
│   │   └── *_viz.png (triptych visualizations)
│   ├── metrics.csv (per-image PSNR/SSIM)
│   └── summary.json (aggregate results)
└── evaluation_results.json (latest evaluation summary)
```

## File Structure

```
├── model_script.py              # Main model implementation
├── cpu_train_config.json      # Training configuration
├── requirements.txt           # Python dependencies
├── checkpoints/              # Model checkpoints
│   ├── best_model.pth        # Best performing model
│   └── checkpoint_epoch_*.pth # Epoch checkpoints
├── results/                  # Evaluation results
│   └── eval_*/               # Timestamped evaluation runs
├── logs/                     # TensorBoard training logs
└── README.md                 # This file
```

## Technical Details

### Conditional Prior Innovation
The key innovation addressing gray output collapse is the conditional prior:
```python
μ(x) = prior_mu(condition_encoder(x))
z = μ(x)  # Instead of z = 0 for deterministic inference
```

### Invertible Architecture
Each coupling layer performs:
```python
x1, x2 = x.chunk(2, dim=1)
scale, shift = coupling_net([x1, condition])
scale = 0.5 * tanh(scale)  # Bounded for stability
x2_new = x2 * exp(scale) + shift
```

## Current Challenges

### Color Loss

One of the current challenges in the enhancement process is the presence of color loss in the enhanced images. This is evident in the results generated by the model, where the enhanced images sometimes fail to preserve the original color fidelity of the input infrared low-light images. Below is an example:

![Triptych Example](results/eval_20250817_133121/images/56_viz.png)

- **Input Image**: Infrared low-light image
- **Enhanced Image**: Enhanced output with noticeable color loss
- **Ground Truth**: Well-exposed target image

The visualization tools in the project generate triptych images (input | enhanced | target) that highlight this issue.

### Future Target

The future goal of this project is to address and recover the color loss observed in the enhanced images. This will involve:

1. **Improved Loss Functions**: Developing advanced loss functions that better preserve color fidelity.
2. **Enhanced Model Architecture**: Modifying the model to incorporate mechanisms for color consistency.
3. **Dataset Augmentation**: Including more diverse infrared low-light images to improve generalization.

By tackling these challenges, the project aims to achieve higher-quality enhancements that maintain both brightness and color fidelity.


