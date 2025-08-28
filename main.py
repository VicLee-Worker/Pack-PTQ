"""
Main script for Vision Transformer quantization using Pack-PTQ method.

This script implements the complete quantization pipeline including:
1. Model loading and transformation
2. Quantization parameter initialization  
3. Adaptive blocks packing
4. Mixed precision (optional)
5. Module reconstruction
6. Final evaluation and saving
"""

import os
import time
import torch
from loguru import logger

from utils.initialize import create_quantized_model
from utils import (
    parse_args,
    seed_all,
    prepare_calibration_data,
    configure_logger
)
from utils.config import (
    parse_args_to_configs,
    create_weight_quant_params,
    create_act_quant_params,
    create_training_kwargs,
    create_log_filename,
    create_model_filename,
    log_configuration
)
from quant.reconstruction import run_full_reconstruction_pipeline


def main():
    """Main quantization pipeline execution."""
    # Parse command-line arguments
    args = parse_args()
    seed_all(args.seed)

    # Parse configurations
    quant_config, training_config, experiment_config = parse_args_to_configs(args)

    # Setup logging
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = create_log_filename(args.model_name, quant_config, timestamp)
    configure_logger(log_file)

    # Log configuration
    log_configuration(quant_config, training_config, args.model_name)
    logger.info(args)

    # Prepare data
    train_loader, test_loader, cali_data, cali_target = prepare_calibration_data(args)

    # Create quantization parameters
    weight_quant_params = create_weight_quant_params(quant_config)
    act_quant_params = create_act_quant_params(quant_config)
    training_kwargs = create_training_kwargs(training_config, cali_data)

    # Create quantized model
    logger.info("Creating quantized model...")
    q_model = create_quantized_model(
        model_name=args.model_name,
        weight_quant_params=weight_quant_params,
        act_quant_params=act_quant_params,
        calibration_data=cali_data,
        batch_size=args.batch_size,
        timestamp=timestamp,
        pretrained_path=args.pretrained_model_path
    )

    # Run reconstruction pipeline
    logger.info("Starting reconstruction pipeline...")
    save_path = f'./trained_models/{create_model_filename(args.model_name, quant_config, timestamp)}'
    
    final_accuracy = run_full_reconstruction_pipeline(
        model=q_model,
        args=args,
        calibration_data=cali_data,
        test_loader=test_loader,
        weight_quant_params=weight_quant_params,
        training_kwargs=training_kwargs,
        save_path=save_path
    )

    logger.info(f"Quantization pipeline completed successfully!")
    logger.info(f"Final accuracy: {final_accuracy:.2f}")
    logger.info(f"Model saved to: {save_path}")


if __name__ == '__main__':
    main()