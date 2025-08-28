from copy import deepcopy
from typing import Union, Optional, Dict, Any
import torch
import torch.nn as nn
from .quant_module import Uniform_Affine_Quantizer, QuantModule, QuantMatMul


class QuantModel(nn.Module):
    """
    A wrapper for quantizing a given model.
    
    Args:
        model: The original model to be quantized
        weight_quant_params: Weight quantization parameters dict, defaults to None
        act_quant_params: Activation quantization parameters dict, defaults to None
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        weight_quant_params: Optional[Dict[str, Any]] = None, 
        act_quant_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # Handle default parameters
        self.weight_quant_params = weight_quant_params or {}
        self.act_quant_params = act_quant_params or {}
        
        self.model = model
        self._first_module = True
        
        # Recursively replace original modules
        self._replace_modules(self.model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantized model.
        
        Args:
            input: Input tensor
            
        Returns:
            Output tensor from the model
        """
        return self.model(input)

    def _replace_modules(self, module: nn.Module) -> None:
        """
        Recursively replace original modules with quantized versions.
        
        Args:
            module: The module to recursively search and replace
        """
        for name, child_module in module.named_children():
            if self._should_quantize_module(child_module):
                quantized_module = self._create_quantized_module(child_module)
                setattr(module, name, quantized_module)
            else:
                self._replace_modules(child_module)

    def _should_quantize_module(self, module: nn.Module) -> bool:
        """Check if a module should be quantized"""
        return isinstance(module, (nn.Conv2d, nn.Linear))

    def _create_quantized_module(self, module: Union[nn.Conv2d, nn.Linear]) -> QuantModule:
        """Create quantized module"""
        if self._first_module:
            quantized_module = QuantModule(
                module, 
                self.weight_quant_params, 
                self.act_quant_params, 
                is_first_module=True
            )
            self._first_module = False
        else:
            quantized_module = QuantModule(
                module, 
                self.weight_quant_params, 
                self.act_quant_params
            )
        return quantized_module

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        """
        Set quantization state for the model.
        
        Args:
            weight_quant: Whether to enable weight quantization, defaults to False
            act_quant: Whether to enable activation quantization, defaults to False
        """
        for module in self.model.modules():
            if isinstance(module, (QuantModule, QuantMatMul)):
                module.set_quant_state(weight_quant, act_quant)

    def set_first_and_last_layer_to_8_bit(self) -> None:
        """Set the first and last layers of the model to 8-bit quantization"""
        weight_quantizers, act_quantizers = self._collect_quantizers()
        
        if weight_quantizers:
            weight_quantizers[0].bitwidth_refactor(8)
            weight_quantizers[-1].bitwidth_refactor(8)
        
        if act_quantizers:
            act_quantizers[0].bitwidth_refactor(8) 
            act_quantizers[-1].bitwidth_refactor(8)

    def _collect_quantizers(self) -> tuple[list, list]:
        """Collect all weight and activation quantizers"""
        weight_quantizers = []
        act_quantizers = []
        
        for module in self.model.modules():
            if isinstance(module, Uniform_Affine_Quantizer):
                if module.is_act:
                    act_quantizers.append(module)
                else:
                    weight_quantizers.append(module)
        
        return weight_quantizers, act_quantizers

    def get_quantization_info(self) -> Dict[str, Any]:
        """
        Get a summary of model quantization information.
        
        Returns:
            Dictionary containing quantization information
        """
        weight_quantizers, act_quantizers = self._collect_quantizers()
        
        info = {
            'total_quantized_modules': len(list(self._get_quantized_modules())),
            'weight_quantizers_count': len(weight_quantizers),
            'act_quantizers_count': len(act_quantizers),
            'weight_bitwidths': [q.n_bits for q in weight_quantizers],
            'act_bitwidths': [q.n_bits for q in act_quantizers]
        }
        
        return info

    def _get_quantized_modules(self):
        """Generator for all quantized modules"""
        for module in self.model.modules():
            if isinstance(module, (QuantModule, QuantMatMul)):
                yield module

    def enable_quantization(self) -> None:
        """Enable all quantization"""
        self.set_quant_state(weight_quant=True, act_quant=True)

    def disable_quantization(self) -> None:
        """Disable all quantization"""
        self.set_quant_state(weight_quant=False, act_quant=False)

    def enable_weight_quantization_only(self) -> None:
        """Enable weight quantization only"""
        self.set_quant_state(weight_quant=True, act_quant=False)

    def enable_activation_quantization_only(self) -> None:
        """Enable activation quantization only"""
        self.set_quant_state(weight_quant=False, act_quant=True)
