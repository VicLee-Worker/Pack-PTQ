"""
Configuration management for quantization experiments.

This module provides functions for creating and managing quantization
configurations and experiment parameters.
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class QuantConfig:
    """Quantization configuration parameters."""
    weight_bits: int = 3
    weight_channel_wise: bool = True
    weight_scale_method: str = 'mse'
    
    activation_bits: int = 3
    activation_channel_wise: bool = False
    activation_scale_method: str = 'mse'
    
    enable_mixed_precision: bool = False


@dataclass
class TrainingConfig:
    """Training and calibration configuration."""
    batch_size: int = 32
    calibration_samples: int = 1024
    calibration_iterations: int = 20000
    learning_rate: float = 4e-5
    reconstruction_weight: float = 0.01
    warmup_ratio: float = 0.2
    
    # Temperature parameters
    temperature_start: int = 20
    temperature_end: int = 2
    
    # Regularization parameters
    input_stochastic_prob: float = 0.5
    rounding_regularization: float = 0.1
    cross_layer_regularization: float = 0.02
    distillation_temperature: float = 4.0
    bn_learning_rate: float = 1e-3


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    model_name: str = 'deit_tiny'
    dataset_path: str = '/media/victor/WorkSpace/Work/ImageNet/'
    seed: int = 1005
    num_workers: int = 4
    pretrained_model_path: Optional[str] = None


def create_weight_quant_params(config: QuantConfig) -> Dict[str, Any]:
    """Create weight quantization parameters dictionary."""
    return {
        'n_bits': config.weight_bits,
        'channel_wise': config.weight_channel_wise,
        'scale_method': config.weight_scale_method,
        'is_act': False
    }


def create_act_quant_params(config: QuantConfig) -> Dict[str, Any]:
    """Create activation quantization parameters dictionary."""
    return {
        'n_bits': config.activation_bits,
        'channel_wise': config.activation_channel_wise,
        'scale_method': config.activation_scale_method,
        'is_act': True
    }


def create_training_kwargs(config: TrainingConfig, cali_data) -> Dict[str, Any]:
    """Create training keyword arguments dictionary."""
    return {
        'cali_data': cali_data,
        'batch_size': config.batch_size,
        'iters': config.calibration_iterations,
        'weight': config.reconstruction_weight,
        'b_range': (config.temperature_start, config.temperature_end),
        'warmup': config.warmup_ratio,
        'opt_mode': 'mse',
        'lr': config.learning_rate,
        'input_prob': config.input_stochastic_prob,
        'lamb_r': config.rounding_regularization,
        'T': config.distillation_temperature,
        'bn_lr': config.bn_learning_rate,
        'lamb_c': config.cross_layer_regularization
    }


def create_log_filename(model: str, quant_config: QuantConfig, timestamp: str = None) -> str:
    """Create log filename based on configuration."""
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    mp_suffix = '_mixed_precision' if quant_config.enable_mixed_precision else ''
    return f"{timestamp}_{model}_w{quant_config.weight_bits}_a{quant_config.activation_bits}{mp_suffix}.log"


def create_model_filename(model: str, quant_config: QuantConfig, timestamp: str = None) -> str:
    """Create model filename based on configuration."""
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    return f"{timestamp}_{model}_{quant_config.weight_bits}_{quant_config.activation_bits}.pt"


def log_configuration(quant_config: QuantConfig, training_config: TrainingConfig, model: str):
    """Log the current configuration settings."""
    from loguru import logger
    
    logger.info(f"Model: {model}")
    logger.info(f"Weight quantization: {quant_config.weight_bits} bits, channel_wise: {quant_config.weight_channel_wise}")
    logger.info(f"Activation quantization: {quant_config.activation_bits} bits, channel_wise: {quant_config.activation_channel_wise}")
    logger.info(f"Mixed precision: {quant_config.enable_mixed_precision}")
    logger.info(f"Batch size: {training_config.batch_size}")
    logger.info(f"Training iterations: {training_config.calibration_iterations}")
    logger.info(f"Learning rate: {training_config.learning_rate}")


def parse_args_to_configs(args):
    """Convert argparse results to configuration objects."""
    quant_config = QuantConfig(
        weight_bits=args.weight_bits,
        weight_channel_wise=args.weight_channel_wise if hasattr(args, 'weight_channel_wise') else True,
        weight_scale_method=args.weight_init_method,
        activation_bits=args.activation_bits,
        activation_channel_wise=args.activation_channel_wise if hasattr(args, 'activation_channel_wise') else False,
        activation_scale_method=args.activation_init_method,
        enable_mixed_precision=args.enable_mixed_precision if hasattr(args, 'enable_mixed_precision') else False
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        calibration_samples=args.calibration_samples,
        calibration_iterations=args.calibration_iterations,
        learning_rate=args.learning_rate,
        reconstruction_weight=args.reconstruction_weight,
        warmup_ratio=args.warmup_ratio,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        input_stochastic_prob=args.input_stochastic_prob,
        rounding_regularization=args.rounding_regularization,
        cross_layer_regularization=args.cross_layer_regularization,
        distillation_temperature=args.distillation_temperature,
        bn_learning_rate=args.bn_learning_rate
    )
    
    experiment_config = ExperimentConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained_model_path=args.pretrained_model_path
    )
    
    return quant_config, training_config, experiment_config
