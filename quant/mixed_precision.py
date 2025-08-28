import torch
import torch.nn as nn
from loguru import logger
from typing import List, Tuple, Dict, Any, Union

from quant.quant_module import AdaRoundQuantizer, QuantMatMul, QuantModule, Uniform_Affine_Quantizer


class ParameterCounter:
    """Utility class for counting quantization parameters."""
    
    @staticmethod
    def count_module_params(module: Union[torch.nn.Module, List]) -> int:
        """Count quantization parameters in a module or list of modules."""
        if isinstance(module, torch.nn.Module):
            return ParameterCounter._count_single_module_params(module)
        else:
            return sum(ParameterCounter.count_module_params(item[1]) for item in module)
    
    @staticmethod
    def _count_single_module_params(module: torch.nn.Module) -> int:
        """Count parameters in a single module."""
        params_count = 0
        for submodule in module.modules():
            if isinstance(submodule, (QuantModule, QuantMatMul)):
                params_count += submodule.act_num
                if isinstance(submodule, QuantModule):
                    params_count += submodule.weight_num
        return params_count


class BitWidthManager:
    """Manages bit width allocation and configuration."""
    
    def __init__(self, target_bits: int):
        self.target_bits = target_bits
        self.bit_choices = self._determine_bit_choices()
    
    def _determine_bit_choices(self) -> List[int]:
        """Determine available bit choices based on target bits."""
        bit_choices_map = {
            3: [2, 3, 4],
            4: [3, 4, 6],
            6: [4, 6, 8]
        }
        
        if self.target_bits >= 8:
            raise ValueError("Only support bit widths less than 8.")
        
        if self.target_bits not in bit_choices_map:
            raise ValueError(f"Unsupported bit width: {self.target_bits}")
            
        return bit_choices_map[self.target_bits]
    
    def allocate_bits_to_packs(self, packs_with_scores: List[Tuple[int, Any]], 
                               storage_limit: int) -> Dict[int, int]:
        """
        Allocate bits to packs based on storage constraints.
        
        Args:
            packs_with_scores: List of (index, pack) tuples
            storage_limit: Maximum storage allowed
            
        Returns:
            Dictionary mapping pack index to allocated bits
        """
        present_bit = self.bit_choices[1]  # Middle bit choice
        high_bit = self.bit_choices[2]     # Highest bit choice
        
        # Initialize all packs with present bit
        allocation = {index: present_bit for index, _ in packs_with_scores}
        
        # Try to upgrade packs to higher bits if storage allows
        for index, _ in packs_with_scores:
            allocation[index] = high_bit
            if self._calculate_total_storage(allocation, packs_with_scores) <= storage_limit:
                continue
            allocation[index] = present_bit  # Revert if exceeds limit
        
        return allocation
    
    def _calculate_total_storage(self, allocation: Dict[int, int], 
                               packs_with_scores: List[Tuple[int, Any]]) -> int:
        """Calculate total storage for given allocation."""
        total_storage = 0
        for index, pack in packs_with_scores:
            pack_params = sum(
                ParameterCounter._count_single_module_params(block[1])
                for block in pack
            )
            total_storage += pack_params * allocation[index]
        return total_storage


class QuantizerConfigurator:
    """Handles quantizer configuration and bit setting."""
    
    @staticmethod
    def set_module_bits(modules: List, bits: int, weight_channel_wise: bool = False, 
                       weight_init_method: str = "max"):
        """Set bit width for a list of modules."""
        for module_name, module in modules:
            QuantizerConfigurator._configure_single_module(
                module, bits, weight_channel_wise, weight_init_method
            )
    
    @staticmethod
    def _configure_single_module(module: torch.nn.Module, bits: int, 
                               weight_channel_wise: bool, weight_init_method: str):
        """Configure quantizers for a single module."""
        for submodule in module.modules():
            if isinstance(submodule, QuantModule):
                submodule.weight_quantizer = Uniform_Affine_Quantizer(
                    bits, weight_channel_wise, weight_init_method
                )
                if not isinstance(submodule.act_quantizer, torch.nn.Identity):
                    submodule.act_quantizer.bitwidth_refactor(bits)
            elif isinstance(submodule, QuantMatMul):
                submodule.quantizer_A.bitwidth_refactor(bits)
                submodule.quantizer_B.bitwidth_refactor(bits)
    
    @staticmethod
    def set_pack_bits(pack: List, bits: int):
        """Set bit width for a pack (list of blocks)."""
        for block in pack:
            for module in block[1].modules():
                if isinstance(module, Uniform_Affine_Quantizer):
                    module.bitwidth_refactor(bits)
    
    @staticmethod
    def reinitialize_quantization_params(model, cali_data: torch.Tensor, batch_size: int):
        """Reinitialize quantization parameters using calibration data."""
        logger.info("Reinitializing quantization parameters...")
        
        # Reset all quantizer initialization flags
        QuantizerConfigurator._reset_quantizer_flags(model, initialized=False)
        
        # Run one batch through the model to initialize parameters
        model.set_quant_state(True, True)
        with torch.no_grad():
            batch_data = cali_data[:batch_size].cuda()
            model(batch_data)
        
        torch.cuda.empty_cache()
        
        # Set all quantizers as initialized
        QuantizerConfigurator._reset_quantizer_flags(model, initialized=True)
        
        logger.info("Quantization parameter reinitialization completed.")
    
    @staticmethod
    def _reset_quantizer_flags(model, initialized: bool):
        """Reset quantizer initialization flags."""
        for name, module in model.named_modules():
            if isinstance(module, QuantModule):
                module.weight_quantizer.params_inited = initialized
                if hasattr(module.act_quantizer, "params_inited"):
                    module.act_quantizer.params_inited = initialized
            elif isinstance(module, QuantMatMul):
                module.quantizer_A.params_inited = initialized
                module.quantizer_B.params_inited = initialized


def count_params(input_module):
    """Legacy function - kept for backward compatibility."""
    return ParameterCounter.count_module_params(input_module)


def determine_bit_choices(args):
    """Legacy function - kept for backward compatibility."""
    if args.weight_bits != args.activation_bits:
        raise ValueError("Only support activation and weight quantization equal.")
    return BitWidthManager(args.weight_bits)._determine_bit_choices()


def calculate_all_pack_storage(allocation, modules):
    """Legacy function - kept for backward compatibility."""
    bit_manager = BitWidthManager(4)  # Default value, not used in legacy calculation
    return bit_manager._calculate_total_storage(allocation, modules)


def allocate_bits(modules, bit_choices, total_storage_limit):
    """Legacy function - kept for backward compatibility."""
    bit_manager = BitWidthManager(4)  # Temporary, actual choices passed as parameter
    bit_manager.bit_choices = bit_choices
    return bit_manager.allocate_bits_to_packs(modules, total_storage_limit)


def set_pack_bits(pack, bits):
    """Legacy function - kept for backward compatibility."""
    QuantizerConfigurator.set_pack_bits(pack, bits)

class MixedPrecisionManager:
    """Main class for managing mixed precision quantization."""
    
    def __init__(self, args, q_model):
        self.args = args
        self.q_model = q_model
        self.bit_manager = BitWidthManager(args.weight_bits)
        self.configurator = QuantizerConfigurator()
        self.param_counter = ParameterCounter()
    
    def apply_mixed_precision(self, wq_params: Dict, cali_data: torch.Tensor,
                            blocks_packing_list: List, blocks_packing_list_score: List,
                            before_blocks_list: List, after_blocks_list: List, **kwargs):
        """
        Apply mixed precision quantization to the model.
        
        Args:
            wq_params: Weight quantization parameters
            cali_data: Calibration data
            blocks_packing_list: List of block packs
            blocks_packing_list_score: Scores for each pack
            before_blocks_list: Modules before blocks
            after_blocks_list: Modules after blocks
            **kwargs: Additional arguments
        """
        logger.info("Mixed precision mode enabled. Assigning different bit-widths to each pack...")
        
        # Configure before and after blocks with highest bit width
        self._configure_boundary_modules(before_blocks_list + after_blocks_list)
        
        # Allocate bits to packs based on scores and constraints
        allocated_bits = self._allocate_pack_bits(
            blocks_packing_list, blocks_packing_list_score, 
            before_blocks_list, after_blocks_list
        )
        
        logger.info(f"Allocated bits for each pack: {allocated_bits}")
        
        # Update quantizers with allocated bits
        self._update_quantizers(wq_params, blocks_packing_list, allocated_bits)
        
        # Reinitialize quantization parameters
        self.configurator.reinitialize_quantization_params(
            self.q_model, cali_data, self.args.batch_size
        )
    
    def _configure_boundary_modules(self, boundary_modules: List):
        """Configure modules before and after blocks with highest bit width."""
        high_bit = self.bit_manager.bit_choices[2]
        self.configurator.set_module_bits(
            boundary_modules, high_bit, 
            self.args.weight_channel_wise, self.args.weight_init_method
        )
    
    def _allocate_pack_bits(self, blocks_packing_list: List, blocks_packing_list_score: List,
                           before_blocks_list: List, after_blocks_list: List) -> Dict[int, int]:
        """Allocate bit widths to packs based on scores and constraints."""
        # Sort packs by score (higher score gets higher priority)
        combined_packs = list(enumerate(zip(blocks_packing_list_score, blocks_packing_list)))
        sorted_packs = [
            (index, block) for index, (score, block) in 
            sorted(combined_packs, key=lambda x: x[1][0], reverse=True)
        ]
        
        # Calculate memory constraints
        total_model_params = self.param_counter.count_module_params(self.q_model)
        boundary_params = self.param_counter.count_module_params(
            before_blocks_list + after_blocks_list
        )
        
        # Set memory constraint (average of mid and high bit widths for total model)
        model_mem_constraint = total_model_params * (
            self.bit_manager.bit_choices[1] + self.bit_manager.bit_choices[2]
        ) * 0.5
        
        # Subtract boundary modules memory (they use high bits)
        packs_mem_constraint = (
            model_mem_constraint - 
            boundary_params * self.bit_manager.bit_choices[2]
        )
        
        # Allocate bits within constraints
        allocated_bits = self.bit_manager.allocate_bits_to_packs(
            sorted_packs, packs_mem_constraint
        )
        
        # Sort allocation by original pack indices
        return dict(sorted(allocated_bits.items()))
    
    def _update_quantizers(self, wq_params: Dict, blocks_packing_list: List, 
                          allocated_bits: Dict[int, int]):
        """Update quantizers with allocated bit widths."""
        # Replace AdaRound quantizers with Uniform quantizers
        for module in self.q_model.modules():
            if isinstance(module, AdaRoundQuantizer):
                # This seems to be a bug in original code - module is not actually replaced
                # Keeping for compatibility but this should be investigated
                module = Uniform_Affine_Quantizer(**wq_params)
        
        # Set bit widths for each pack
        for index, bits in allocated_bits.items():
            self.configurator.set_pack_bits(blocks_packing_list[index], bits)


def mixed_precision_mode(args, wq_params, q_model, cali_data, kwargs, 
                        blocks_packing_list, blocks_packing_list_score, 
                        before_blocks_list, after_blocks_list):
    """
    Main function for mixed precision mode - refactored for better maintainability.
    
    Args:
        args: Command-line arguments.
        wq_params: Weight quantization parameters.
        q_model: The quantized model.
        cali_data: Calibration data.
        kwargs: Additional keyword arguments.
        blocks_packing_list: The blocks packing list.
        blocks_packing_list_score: The blocks packing list scores.
        before_blocks_list: The list of modules before blocks.
        after_blocks_list: The list of modules after blocks.
    """
    manager = MixedPrecisionManager(args, q_model)
    # Remove cali_data from training_kwargs if it exists to avoid parameter conflicts
    clean_kwargs = {k: v for k, v in kwargs.items() if k != 'cali_data'}
    manager.apply_mixed_precision(
        wq_params, cali_data, blocks_packing_list, blocks_packing_list_score,
        before_blocks_list, after_blocks_list, **clean_kwargs
    )