import os
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Input/Output dimensions
    input_size: Tuple[int, int] = (256, 256)
    channels: int = 3
    
    # Encoder configuration
    encoder_channels: list = None
    encoder_kernel_sizes: list = None
    encoder_strides: list = None
    
    # Decoder configuration  
    decoder_channels: list = None
    decoder_kernel_sizes: list = None
    decoder_strides: list = None
    
    # Flow configuration
    num_flow_blocks: int = 4
    flow_channels: int = 256
    
    # GAN configuration
    discriminator_channels: list = None
    discriminator_kernel_sizes: list = None
    discriminator_strides: list = None
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [64, 128, 256]
        if self.encoder_kernel_sizes is None:
            self.encoder_kernel_sizes = [3, 3, 3]
        if self.encoder_strides is None:
            self.encoder_strides = [2, 2, 2]
            
        if self.decoder_channels is None:
            self.decoder_channels = [128, 64, 3]
        if self.decoder_kernel_sizes is None:
            self.decoder_kernel_sizes = [3, 3, 3]
        if self.decoder_strides is None:
            self.decoder_strides = [2, 2, 2]
            
        if self.discriminator_channels is None:
            self.discriminator_channels = [64, 128, 256, 512]
        if self.discriminator_kernel_sizes is None:
            self.discriminator_kernel_sizes = [4, 4, 4, 4]
        if self.discriminator_strides is None:
            self.discriminator_strides = [2, 2, 2, 2]

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data paths
    train_low_dir: str = "train-20241116T084056Z-001-20250713T124140Z-1-001/train-20241116T084056Z-001/train/low"
    train_high_dir: str = "train-20241116T084056Z-001-20250713T124140Z-1-001/train-20241116T084056Z-001/train/high"
    eval_low_dir: str = "eval-20241116T084057Z-001-20250713T124140Z-1-001/eval-20241116T084057Z-001/eval/low"
    eval_high_dir: str = "eval-20241116T084057Z-001-20250713T124140Z-1-001/eval-20241116T084057Z-001/eval/high"
    
    # Training parameters
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Loss weights
    content_loss_weight: float = 1.0
    adversarial_loss_weight: float = 0.1
    perceptual_loss_weight: float = 0.1
    color_loss_weight: float = 0.01
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Validation
    val_frequency: int = 5
    save_frequency: int = 10
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    model_name: str = "low_light_enhancer"
    
    # Logging
    log_dir: str = "logs"
    use_wandb: bool = True
    wandb_project: str = "low-light-enhancement"

@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_path: str = "checkpoints/best_model.pth"
    output_dir: str = "results"
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Post-processing
    use_post_processing: bool = True
    gamma_correction: float = 1.2
    contrast_enhancement: bool = True
    
    # Batch inference
    batch_size: int = 1
    
    # Output format
    output_format: str = "png"  # "png", "jpg"
    quality: int = 95

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics
    metrics: list = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["psnr", "ssim", "lpips", "fid"]
    
    # Dataset
    test_low_dir: str = "eval-20241116T084057Z-001-20250713T124140Z-1-001/eval-20241116T084057Z-001/eval/low"
    test_high_dir: str = "eval-20241116T084057Z-001-20250713T124140Z-1-001/eval-20241116T084057Z-001/eval/high"
    
    # Evaluation parameters
    batch_size: int = 1
    save_results: bool = True
    results_dir: str = "evaluation_results"

# Global configurations
model_config = ModelConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
evaluation_config = EvaluationConfig()

# Create directories
os.makedirs(training_config.checkpoint_dir, exist_ok=True)
os.makedirs(training_config.log_dir, exist_ok=True)
os.makedirs(inference_config.output_dir, exist_ok=True)
os.makedirs(evaluation_config.results_dir, exist_ok=True) 