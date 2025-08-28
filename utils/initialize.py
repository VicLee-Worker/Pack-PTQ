"""
Model creation, attention modification, and quantization initialization utilities.

This module consolidates the former functionality of `models/transformer_quantizer.py`:
- Base FP model creation
- Attention layer patching for ViT/DeiT/Swin with quantized matmuls
- Quantization model construction and parameter initialization
- Model (state dict) saving/loading helpers
"""

import os
from typing import Optional
import timm
import torch
from loguru import logger
from types import MethodType
from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention
# from timm.layers.attention import maybe_add_mask
import torch.nn.functional as F

from quant import QuantModel, QuantModule, QuantMatMul
from utils import hubconf


# ------------------------------
# Attention forward replacements
# ------------------------------

def vit_attention_forward(self, x):
    """Quantized forward pass for Vision Transformer attention."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # Use quantized matrix multiplication
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# def vit_attention_forward(
#         self,
#         x: torch.Tensor,
#         attn_mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     B, N, C = x.shape
#     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#     q, k, v = qkv.unbind(0)
#     q, k = self.q_norm(q), self.k_norm(k)

#     if self.fused_attn:
#         x = F.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=attn_mask,
#             dropout_p=self.attn_drop.p if self.training else 0.,
#         )
#     else:
#         q = q * self.scale
#         attn = self.matmul1(q, k.transpose(-2, -1))
#         # attn = q @ k.transpose(-2, -1)
#         attn = maybe_add_mask(attn, attn_mask)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = self.matmul2(attn, v)
#         # x = attn @ v

#     x = x.transpose(1, 2).reshape(B, N, C)
#     x = self.norm(x)
#     x = self.proj(x)
#     x = self.proj_drop(x)
#     return x


def swin_attention_forward(self, x, mask=None):
    """Quantized forward pass for Swin Transformer attention."""
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1))

    # Add relative position bias
    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
    ].view(
        self.window_size[0] * self.window_size[1], 
        self.window_size[0] * self.window_size[1], 
        -1
    )
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    attn = attn + relative_position_bias.unsqueeze(0)

    # Apply mask if provided
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# def swin_attention_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#     """Forward pass.

#     Args:
#         x: Input features with shape of (num_windows*B, N, C).
#         mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None.

#     Returns:
#         Output features with shape of (num_windows*B, N, C).
#     """
#     B_, N, C = x.shape
#     qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#     q, k, v = qkv.unbind(0)

#     if self.fused_attn:
#         attn_mask = self._get_rel_pos_bias()
#         if mask is not None:
#             num_win = mask.shape[0]
#             mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
#             attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
#         x = torch.nn.functional.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=attn_mask,
#             dropout_p=self.attn_drop.p if self.training else 0.,
#         )
#     else:
#         q = q * self.scale
#         attn = self.matmul1(q, k.transpose(-2, -1))
#         # attn = q @ k.transpose(-2, -1)
#         attn = attn + self._get_rel_pos_bias()
#         if mask is not None:
#             num_win = mask.shape[0]
#             attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#         attn = self.softmax(attn)
#         attn = self.attn_drop(attn)
#         x = self.matmul2(attn, v)
#         # x = attn @ v

#     x = x.transpose(1, 2).reshape(B_, N, -1)
#     x = self.proj(x)
#     x = self.proj_drop(x)
    # return x


# ------------------------------
# Model creation and modification
# ------------------------------

def create_base_model(model_name: str):
    """Create base floating-point model."""
    model_zoo = {
        'vit_tiny': 'vit_tiny_patch16_224',
        'vit_small': 'vit_small_patch16_224',
        'vit_base': 'vit_base_patch16_224',
        'deit_tiny': 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base': 'deit_base_patch16_224',
        'swin_tiny': 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base': 'swin_base_patch4_window7_224',
    }

    if 'net' in model_name:
        return eval(f'hubconf.{model_name}(pretrained=True)')
    else:
        return timm.create_model(model_zoo[model_name], pretrained=True)


def modify_attention_layers(model, model_name: str, act_quant_params: dict):
    """Modify attention layers for quantization."""
    if 'vit' in model_name or 'deit' in model_name:
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                setattr(module, "matmul1", QuantMatMul(act_quant_params))
                act_quant_params['log_quant'] = True
                setattr(module, "matmul2", QuantMatMul(act_quant_params))
                module.forward = MethodType(vit_attention_forward, module)
    
    elif 'swin' in model_name:
        for name, module in model.named_modules():
            if isinstance(module, WindowAttention):
                setattr(module, "matmul1", QuantMatMul(act_quant_params))
                setattr(module, "matmul2", QuantMatMul(act_quant_params))
                module.forward = MethodType(swin_attention_forward, module)


def _set_param_by_name(root_module: torch.nn.Module, name: str, tensor: torch.Tensor):
    """Set a parameter/buffer by dotted path name on a module, creating an nn.Parameter of given tensor.
    Supports indexing by numeric child (e.g., sequential indices)."""
    parts = name.split('.')
    module = root_module
    for p in parts[:-1]:
        module = module[int(p)] if p.isdigit() else getattr(module, p)
    setattr(module, parts[-1], torch.nn.Parameter(tensor))


def initialize_quantization_params(model, calibration_data, batch_size: int):
    """Initialize quantization parameters using calibration data."""
    logger.info("Initializing quantization parameters...")
    
    model.set_quant_state(True, True)
    
    # Reset initialization flags
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.params_inited = False
            if hasattr(module.act_quantizer, "params_inited"):
                module.act_quantizer.params_inited = False
        if isinstance(module, QuantMatMul):
            module.quantizer_A.params_inited = False
            module.quantizer_B.params_inited = False

    # Forward pass with calibration data
    with torch.no_grad():
        for i in range(int(calibration_data.size(0) / batch_size)):
            model(calibration_data[i * batch_size:(i + 1) * batch_size].cuda())
            break
    
    torch.cuda.empty_cache()

    # Set initialization flags
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.params_inited = True
            if hasattr(module.act_quantizer, "params_inited"):
                module.act_quantizer.params_inited = True
        if isinstance(module, QuantMatMul):
            module.quantizer_A.params_inited = True
            module.quantizer_B.params_inited = True

    logger.info("Quantization parameters initialization completed")


def save_initialized_model(model, timestamp: str, model_name: str, weight_bits: int, activation_bits: int):
    """Save initialized quantization model state dict."""
    os.makedirs('./inited_models/', exist_ok=True)
    save_path = f'./inited_models/{timestamp}_{model_name}_{weight_bits}_{activation_bits}.pt'
    
    # Save only the state dict to reduce file size
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'weight_bits': weight_bits,
        'activation_bits': activation_bits,
        'timestamp': timestamp
    }, save_path)
    
    logger.info(f"Initialized model state dict saved to {save_path}")


def load_pretrained_quant_model(model_path: str, model_name: str, weight_quant_params: dict, act_quant_params: dict):
    """Load pre-trained quantization model from state dict."""
    logger.info(f"Loading initialized model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    fp_model = create_base_model(model_name)
    modify_attention_layers(fp_model, model_name, act_quant_params)
    fp_model.cuda()
    fp_model.eval()

    quant_model = QuantModel(
        model=fp_model,
        weight_quant_params=weight_quant_params,
        act_quant_params=act_quant_params
    )

    model_sd = quant_model.state_dict()
    ckpt_sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    for name, tensor in ckpt_sd.items():
        if name in model_sd and tuple(tensor.shape) != tuple(model_sd[name].shape):
            try:
                _set_param_by_name(quant_model, name, tensor)
            except Exception as e:
                logger.warning(f"Failed to reshape param {name}: {e}")

    missing, unexpected = quant_model.load_state_dict(ckpt_sd, strict=False)
    if missing or unexpected:
        logger.debug(f"State dict loaded with missing={len(missing)}, unexpected={len(unexpected)}")

    quant_model.cuda()
    quant_model.eval()

    # Set quantization parameters as initialized
    for name, module in quant_model.named_modules():
        if hasattr(module, 'weight_quantizer'):
            module.weight_quantizer.params_inited = True
        if hasattr(module, 'act_quantizer') and hasattr(module.act_quantizer, "params_inited"):
            module.act_quantizer.params_inited = True
        if hasattr(module, 'quantizer_A'):
            module.quantizer_A.params_inited = True
        if hasattr(module, 'quantizer_B'):
            module.quantizer_B.params_inited = True

    logger.info("Pre-trained quantization model loaded successfully from state dict.")
    return quant_model


def create_quantized_model(
    model_name: str,
    weight_quant_params: dict,
    act_quant_params: dict,
    calibration_data,
    batch_size: int,
    timestamp: str,
    pretrained_path: str = None
):
    """
    Create and initialize a quantized model.
    
    Args:
        model_name: Name of the model architecture
        weight_quant_params: Weight quantization parameters
        act_quant_params: Activation quantization parameters
        calibration_data: Calibration dataset
        batch_size: Batch size for calibration
        timestamp: Timestamp for saving
        pretrained_path: Path to pre-trained quantized model
        
    Returns:
        Quantized model ready for further processing
    """
    if pretrained_path:
        return load_pretrained_quant_model(
            pretrained_path, 
            model_name, 
            weight_quant_params, 
            act_quant_params
        )
    
    # Create base model
    fp_model = create_base_model(model_name)
    
    # Modify attention layers for quantization
    modify_attention_layers(fp_model, model_name, act_quant_params)
    
    fp_model.cuda()
    fp_model.eval()
    
    logger.info(f"Base model '{model_name}' created and loaded")
    
    # Convert to quantized model
    logger.info("Converting to quantized model...")
    quant_model = QuantModel(
        model=fp_model, 
        weight_quant_params=weight_quant_params, 
        act_quant_params=act_quant_params
    )
    quant_model.cuda()
    quant_model.eval()
    
    # Initialize quantization parameters
    initialize_quantization_params(quant_model, calibration_data, batch_size)
    
    # Save initialized model
    save_initialized_model(
        quant_model, 
        timestamp, 
        model_name, 
        weight_quant_params['n_bits'], 
        act_quant_params['n_bits']
    )
    
    # Clean up
    del fp_model
    
    return quant_model
