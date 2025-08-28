import argparse

def parse_args():
    """
    Parse command-line arguments for Pack-PTQ quantization experiments.

    This function configures and parses all command-line arguments needed for
    post-training quantization of vision models using the Pack-PTQ method.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Pack-PTQ: Advancing Post-training Quantization of Neural Networks by Pack-wise Reconstruction'
    )

    # Dataset and Model Configuration
    parser.add_argument('--seed', type=int, default=1005,
                        help='Random seed for reproducible experiments.')
    parser.add_argument('--dataset_path', type=str, default='/media/victor/WorkSpace/Work/ImageNet/',
                        help='Path to the ImageNet dataset directory.')
    parser.add_argument('--model_name', type=str, default='deit_tiny',
                        choices=['vit_small', 'vit_base',
                                 'deit_tiny', 'deit_small', 'deit_base', 
                                 'swin_tiny', 'swin_small', 'swin_base',
                                 'resnet18', 'resnet50',
                                 'mobilenetv2',
                                 'regnetx_600m', 'regnetx_3200m',
                                 'mnasnet'],
                        help='Model architecture to quantize (supports Vision Transformers and CNN models).')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data loading during calibration and evaluation.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers.')
    parser.add_argument('--calibration_samples', type=int, default=1024,
                        help='Number of calibration samples for quantization parameter initialization.')
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='Path to pre-trained model checkpoint to load.')

    # Quantization Configuration
    parser.add_argument('--weight_bits', type=int, default=3,
                        help='Bit-width for weight quantization.')
    parser.add_argument('--weight_channel_wise', action='store_true',
                        help='Enable channel-wise quantization for weights to improve accuracy.')
    parser.add_argument('--activation_bits', type=int, default=3,
                        help='Bit-width for activation quantization.')
    parser.add_argument('--activation_channel_wise', action='store_true',
                        help='Enable channel-wise quantization for activations.')
    parser.add_argument('--enable_mixed_precision', action='store_true',
                        help='Enable mixed precision quantization with different bit-widths for different layers.')

    # Calibration and Reconstruction Hyperparameters
    parser.add_argument('--calibration_iterations', type=int, default=20000,
                        help='Number of optimization iterations for quantization parameter calibration.')
    parser.add_argument('--reconstruction_weight', type=float, default=0.01,
                        help='Balancing weight between rounding cost and reconstruction loss.')
    parser.add_argument('--temperature_start', type=int, default=20,
                        help='Initial temperature parameter for the calibration annealing schedule.')
    parser.add_argument('--temperature_end', type=int, default=2,
                        help='Final temperature parameter for the calibration annealing schedule.')
    parser.add_argument('--warmup_ratio', type=float, default=0.2,
                        help='Warmup ratio for calibration (fraction of iterations without regularization).')
    parser.add_argument('--weight_init_method', type=str, default='mse',
                        choices=['minmax', 'mse', 'minmax_scale'],
                        help='Initialization method for weight quantization parameters.')
    parser.add_argument('--activation_init_method', type=str, default='mse',
                        choices=['minmax', 'mse', 'minmax_scale'],
                        help='Initialization method for activation quantization parameters.')
    parser.add_argument('--learning_rate', type=float, default=4e-5,
                        help='Learning rate for quantization parameter optimization.')

    # Advanced Regularization and Training Hyperparameters
    parser.add_argument('--stochastic_prob', type=float, default=0.5,
                        help='Probability parameter for stochastic operations during reconstruction.')
    parser.add_argument('--input_stochastic_prob', type=float, default=0.5,
                        help='Input probability for stochastic operations.')
    parser.add_argument('--rounding_regularization', type=float, default=0.1,
                        help='Regularization coefficient for rounding cost in the reconstruction loss.')
    parser.add_argument('--distillation_temperature', type=float, default=4.0,
                        help='Temperature parameter for knowledge distillation in KL divergence calculation.')
    parser.add_argument('--bn_learning_rate', type=float, default=1e-3,
                        help='Learning rate for batch normalization parameter fine-tuning.')
    parser.add_argument('--cross_layer_regularization', type=float, default=0.02,
                        help='Cross-layer error propagation regularization coefficient.')
    
    return parser.parse_args()
