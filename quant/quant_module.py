from typing import Union, Tuple, Optional, Literal
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

# Constants definition
MIN_DELTA = 1e-12
DEFAULT_MSE_ITERATIONS = 80
DEFAULT_MSE_POWER = 2.4

def lp_loss(pred: torch.Tensor, tgt: torch.Tensor, p: float = 2.0, reduction: str = 'none') -> torch.Tensor:
    """
    Compute L_p norm loss function.
    
    Args:
        pred: Predicted tensor
        tgt: Target tensor  
        p: Order of the norm, defaults to 2.0
        reduction: Reduction method, 'none' or other
        
    Returns:
        Computed loss value
    """
    diff = (pred - tgt).abs().pow(p)
    if reduction == 'none':
        return diff.sum(1).mean()
    else:
        return diff.mean()

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Implement Straight-Through Estimator for rounding operation.
    
    Args:
        x: Input tensor
        
    Returns:
        Rounded tensor with Straight-Through Estimator applied
    """
    return (x.round() - x).detach() + x

def floor_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Implement Straight-Through Estimator for floor operation.
    
    Args:
        x: Input tensor
        
    Returns:
        Floored tensor with Straight-Through Estimator applied
    """
    return (x.floor() - x).detach() + x


class BaseQuantizer(ABC, nn.Module):
    """
    Base class for quantizers, defining common interface for all quantizers.
    """
    
    def __init__(self, n_bits: int = 8, **kwargs):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass method, must be implemented by subclasses"""
        pass
        
    def bitwidth_refactor(self, new_bits: int) -> None:
        """Refactor bit width"""
        self.n_bits = new_bits
        self.n_levels = 2 ** new_bits

class LogSqrt2Quantizer(BaseQuantizer):
    """
    Log sqrt 2 quantizer
    Based on asymmetric quantization implementation from https://arxiv.org/abs/1806.08342
    
    Args:
        n_bits: Number of quantization bits, defaults to 8
        channel_wise: Whether to compute scale and zero_point per channel, defaults to False
        scale_method: Scaling method, defaults to 'mse'
        is_act: Whether it's an activation quantizer, defaults to False
        prob: Quantization probability, defaults to 1.0
    """
    
    def __init__(
        self, 
        n_bits: int = 8, 
        channel_wise: bool = False, 
        scale_method: str = 'mse',
        is_act: bool = False, 
        prob: float = 1.0
    ):
        super().__init__(n_bits)
        if not (2 <= n_bits <= 8):
            raise ValueError(f'Bitwidth {n_bits} not supported, must be between 2 and 8')
            
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.is_act = is_act
        self.prob = prob
        
        self.delta = nn.Parameter(torch.tensor(1.0))
        self.params_inited = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for quantization"""
        if not self.params_inited:
            self.delta = self._init_quantization_scale(x)
            self.params_inited = True

        return self._quantize(x, self.delta)

    def _init_quantization_scale(self, x: torch.Tensor) -> nn.Parameter:
        """Initialize quantization scale"""
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = float('inf')
        
        percentiles = [0.999, 0.9999, 0.99999]
        for pct in percentiles:
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(
                    np.percentile(x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32
                )
            
            x_q = self._quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            
            if score < best_score:
                best_score = score
                delta = new_delta

        return nn.Parameter(delta.clone())

    def _quantize(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Perform quantization operation"""
        from math import sqrt
        
        x_int = torch.round(-1 * (x / delta).log2() * 2)
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        
        odd_mask = (x_quant % 2) * (sqrt(2) - 1) + 1
        x_float_q = 2 ** (-1 * torch.ceil(x_quant / 2)) * odd_mask * delta
        x_float_q[mask] = 0

        return x_float_q


class Uniform_Affine_Quantizer(BaseQuantizer):
    """
    Uniform affine quantizer (also called asymmetric quantization).
    Quantizes arguments in forward pass, passes gradients straight through
    in backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342
    
    Args:
        n_bits: Number of quantization bits, defaults to 8
        channel_wise: Whether to compute scale and zero_point per channel, defaults to False
        scale_method: Quantization scale and zero point determination method, defaults to 'mse'
        is_act: Whether it's an activation quantizer, defaults to False
        prob: Qdrop probability, defaults to 1.0
        parallel_channel: Whether to compute channel scales in parallel, defaults to True
        sym: Whether to use symmetric range in 'max' methods, defaults to False
        log_quant: Whether to use log quantization (for compatibility), defaults to False
        **kwargs: Additional keyword arguments for flexibility
    """
    
    def __init__(
        self, 
        n_bits: int = 8, 
        channel_wise: bool = False, 
        scale_method: str = 'mse',
        is_act: bool = False, 
        prob: float = 1.0, 
        parallel_channel: bool = True, 
        sym: bool = False,
        log_quant: bool = False,  # Add log_quant parameter for compatibility
        **kwargs  # Accept any other parameters for flexibility
    ):
        super().__init__(n_bits)
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.is_act = is_act
        self.prob = prob
        self.parallel_channel = parallel_channel
        self.sym = sym
        self.log_quant = log_quant  # Store log_quant parameter but useless

        self.params_inited = False
        self.n_range = 2 ** self.n_bits
        self.delta = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for quantization"""
        if not self.params_inited:
            delta, zero_point = self._init_quantization_scale(x.clone().detach(), self.channel_wise)
            self.delta.data = delta
            self.zero_point.data = zero_point
            self.params_inited = True

        # Quantization and dequantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_range - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        # Qdrop processing
        if self.prob < 1.0:
            return torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        return x_dequant

    def _quantize_simple(self, x: torch.Tensor, x_max: torch.Tensor, x_min: torch.Tensor) -> torch.Tensor:
        """Simple quantization operation"""
        delta = (x_max - x_min) / (2 ** self.n_bits - 1)
        zero_point = (-x_min / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_range - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def _get_channel_axis(self, x: torch.Tensor) -> int:
        """Get channel axis index"""
        ndim = x.dim()
        if self.is_act:
            if ndim == 4:
                return 1
            elif ndim == 3:
                return 2
            elif ndim == 2:
                return 1
            else:
                raise NotImplementedError(f"Unsupported tensor dimension: {ndim}")
        else:
            if ndim == 4:
                return 0
            elif ndim == 3:
                return 2
            elif ndim == 2:
                return 0
            else:
                raise NotImplementedError(f"Unsupported tensor dimension: {ndim}")

    def _init_channelwise_parallel(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel (vectorized) per-channel scale initialization"""
        xp = x.clone().detach()
        ndim = xp.dim()
        ch_axis = self._get_channel_axis(xp)
        
        # Move channel to first dimension and flatten others
        perm = [ch_axis] + [i for i in range(ndim) if i != ch_axis]
        x_perm = xp.permute(perm).contiguous()
        C = x_perm.shape[0]
        x_cf = x_perm.view(C, -1)
        device, dtype = x_cf.device, x_cf.dtype

        if 'max' in self.scale_method:
            delta_vec, zp_vec = self._init_max_method_parallel(x_cf, device, dtype)
        elif self.scale_method == 'mse':
            delta_vec, zp_vec = self._init_mse_method_parallel(x_cf, device, dtype)
        else:
            raise NotImplementedError(f"Scale method '{self.scale_method}' not implemented")

        # Reshape to broadcast shape
        bshape = [1] * ndim
        bshape[ch_axis] = -1
        delta = delta_vec.view(*bshape)
        zero_point = zp_vec.view(*bshape)
        return delta.clone(), zero_point.clone()

    def _init_max_method_parallel(self, x_cf: torch.Tensor, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel max method initialization"""
        C = x_cf.shape[0]
        zeros = torch.zeros(C, device=device, dtype=dtype)
        x_min = torch.minimum(x_cf.min(dim=1)[0], zeros)
        x_max = torch.maximum(x_cf.max(dim=1)[0], zeros)
        
        if 'scale' in self.scale_method:
            factor = (self.n_bits + 2) / 8
            x_min = x_min * factor
            x_max = x_max * factor
            
        x_absmax = torch.maximum(x_min.abs(), x_max)
        if self.sym:
            x_min = torch.where(x_min < 0, -x_absmax, torch.zeros_like(x_absmax))
            x_max = x_absmax
            
        delta_vec = (x_max - x_min) / (self.n_range - 1)
        # delta_vec = torch.clamp(delta_vec, min=MIN_DELTA)
        zp_vec = torch.round(-x_min / delta_vec)
        return delta_vec, zp_vec

    def _init_mse_method_parallel(self, x_cf: torch.Tensor, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel MSE method initialization"""
        C = x_cf.shape[0]
        x_max0 = x_cf.max(dim=1)[0]
        x_min0 = x_cf.min(dim=1)[0]
        
        best_score = torch.full((C,), float('inf'), device=device, dtype=dtype)
        best_delta = torch.empty(C, device=device, dtype=dtype)
        best_zp = torch.empty(C, device=device, dtype=dtype)
        
        n_levels = 2 ** self.n_bits - 1
        
        for i in range(DEFAULT_MSE_ITERATIONS):
            s = 1.0 - i * 0.01
            new_max = x_max0 * s
            new_min = x_min0 * s
            
            delta_vec = (new_max - new_min) / n_levels
            # delta_vec = torch.clamp(delta_vec, min=MIN_DELTA)
            zp_vec = torch.round(-new_min / delta_vec)
            
            x_int = torch.round(x_cf / delta_vec[:, None])
            x_quant = torch.clamp(x_int + zp_vec[:, None], 0, self.n_range - 1)
            x_deq = (x_quant - zp_vec[:, None]) * delta_vec[:, None]
            
            score = (x_cf - x_deq).abs().pow(DEFAULT_MSE_POWER).mean(dim=1)
            better = score < best_score
            best_score = torch.where(better, score, best_score)
            best_delta = torch.where(better, delta_vec, best_delta)
            best_zp = torch.where(better, zp_vec, best_zp)
            
        return best_delta, best_zp

    def _init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize quantization scale and zero point.
        
        Args:
            x: Input tensor
            channel_wise: Whether to compute per channel
            
        Returns:
            Initialized delta and zero_point tensors
        """
        if channel_wise:
            if self.parallel_channel:
                return self._init_channelwise_parallel(x)
            else:
                return self._init_channelwise_sequential(x)
        else:
            return self._init_tensorwise(x)

    def _init_channelwise_sequential(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential per-channel initialization (maintaining original compatibility)"""
        x_clone = x.clone().detach()
        shape_info = self._get_channel_shape_info(x)
        n_channels, delta_shape, zero_point_shape = shape_info
        
        delta = torch.zeros(n_channels, device=x.device, dtype=x.dtype)
        zero_point = torch.zeros(n_channels, device=x.device, dtype=x.dtype)
        
        for c in range(n_channels):
            if self.is_act:
                x_channel = self._extract_act_channel(x_clone, c, len(x.shape))
            else:
                x_channel = self._extract_weight_channel(x_clone, c, len(x.shape))
                
            delta[c], zero_point[c] = self._init_tensorwise(x_channel)
        
        delta = delta.view(*delta_shape)
        zero_point = zero_point.view(*zero_point_shape)
        return delta, zero_point

    def _get_channel_shape_info(self, x: torch.Tensor) -> Tuple[int, list, list]:
        """Get channel shape information"""
        ndim = len(x.shape)
        if self.is_act:
            n_channels = x.shape[-1] if ndim == 3 else x.shape[1]
            if ndim == 4:
                shape = (1, -1, 1, 1)
            elif ndim == 2:
                shape = (1, -1)
            elif ndim == 3:
                shape = (1, 1, -1)
            else:
                raise NotImplementedError(f"Unsupported activation tensor dimension: {ndim}")
        else:
            n_channels = x.shape[-1] if ndim == 3 else x.shape[0]
            if ndim == 4:
                shape = (-1, 1, 1, 1)
            elif ndim == 2:
                shape = (-1, 1)
            elif ndim == 3:
                shape = (1, 1, -1)
            else:
                raise NotImplementedError(f"Unsupported weight tensor dimension: {ndim}")
        
        return n_channels, list(shape), list(shape)

    def _extract_act_channel(self, x: torch.Tensor, channel: int, ndim: int) -> torch.Tensor:
        """Extract specified channel from activation tensor"""
        if ndim == 3:
            return x[:, :, channel]
        elif ndim == 4:
            return x[:, channel, ...]
        elif ndim == 2:
            return x[:, channel]
        else:
            raise NotImplementedError(f"Unsupported dimension: {ndim}")

    def _extract_weight_channel(self, x: torch.Tensor, channel: int, ndim: int) -> torch.Tensor:
        """Extract specified channel from weight tensor"""
        if ndim == 3:
            return x[:, :, channel]
        elif ndim == 2:
            return x[:, channel]
        else:
            return x[channel]

    def _init_tensorwise(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tensor-wise initialization"""
        if 'max' in self.scale_method:
            return self._init_max_method_tensorwise(x)
        elif self.scale_method == 'mse':
            return self._init_mse_method_tensorwise(x)
        else:
            raise NotImplementedError(f"Scale method '{self.scale_method}' not implemented")

    def _init_max_method_tensorwise(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tensor-wise max method initialization"""
        x_min = min(x.min().item(), 0)
        x_max = max(x.max().item(), 0)
        
        if 'scale' in self.scale_method:
            factor = (self.n_bits + 2) / 8
            x_min = x_min * factor
            x_max = x_max * factor

        x_absmax = max(abs(x_min), x_max)
        if self.sym:
            x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

        delta = float(x_max - x_min) / (self.n_range - 1)
        # if delta < MIN_DELTA:
        #     warnings.warn(f'Quantization range close to zero: [{x_min}, {x_max}]')
        #     delta = MIN_DELTA

        zero_point = round(-x_min / delta)
        return torch.tensor(delta, dtype=x.dtype), torch.tensor(zero_point, dtype=x.dtype)

    def _init_mse_method_tensorwise(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tensor-wise MSE method initialization"""
        x_max = x.max()
        x_min = x.min()
        best_score = float('inf')
        best_delta = None
        best_zero_point = None
        
        for i in range(DEFAULT_MSE_ITERATIONS):
            new_max = x_max * (1.0 - (i * 0.01))
            new_min = x_min * (1.0 - (i * 0.01))
            x_q = self._quantize_simple(x, new_max, new_min)
            score = lp_loss(x, x_q, p=DEFAULT_MSE_POWER, reduction='all')
            
            if score < best_score:
                best_score = score
                best_delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                best_zero_point = (-new_min / best_delta).round()
        
        return best_delta, best_zero_point

    def bitwidth_refactor(self, refactored_bit: int) -> None:
        """Refactor bit width"""
        super().bitwidth_refactor(refactored_bit)
        self.n_range = 2 ** self.n_bits

class AdaRoundQuantizer(BaseQuantizer):
    """
    Adaptive rounding quantizer, optimizes rounding policy
    by reconstructing intermediate output.
    Based on: Up or Down? Adaptive Rounding for Post-Training Quantization
    https://arxiv.org/abs/2004.10568

    Args:
        uaq: Uniform affine quantizer used to initialize quantization parameters
        weight_tensor: Weight tensor for initializing alpha
        round_mode: Rounding mode controlling forward pass, defaults to 'learned_round_sigmoid'
    """

    def __init__(
        self, 
        uaq: Uniform_Affine_Quantizer, 
        weight_tensor: torch.Tensor, 
        round_mode: str = 'learned_round_sigmoid'
    ):
        super().__init__(uaq.n_bits)
        
        # Copy attributes from Uniform_Affine_Quantizer
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_range = uaq.n_range

        self.round_mode = round_mode
        self.alpha: Optional[nn.Parameter] = None
        self.soft_targets = False

        # Sigmoid function parameters
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        
        self._init_alpha(weight_tensor.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for quantization"""
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.delta)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.delta)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor
            x_int = x_floor + torch.bernoulli(rest)
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = floor_ste(x / self.delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError(f'Unknown rounding mode: {self.round_mode}')

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_range - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta
        return x_float_q

    def get_soft_targets(self) -> torch.Tensor:
        """Get soft targets for sigmoid function"""
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 
            0, 1
        )

    def _init_alpha(self, x: torch.Tensor) -> None:
        """Initialize alpha parameter for sigmoid function"""
        x_floor = torch.floor(x / self.delta)
        
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / self.delta) - x_floor  # Rounding remainder [0, 1)
            # Make alpha satisfy sigmoid(alpha) = rest
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError(f"Round mode '{self.round_mode}' not implemented for alpha initialization")

    def extra_repr(self) -> str:
        """Return string representation of the quantizer"""
        return f'bit={self.n_bits}, round_mode={self.round_mode}'


class QuantModule(nn.Module):
    """
    A module that wraps a given nn.Conv2d or nn.Linear module and applies 
    quantization to its weights and activations.
    
    Args:
        org_module: The original module to be quantized
        weight_quant_params: Weight quantization parameters dict, defaults to empty dict
        act_quant_params: Activation quantization parameters dict, defaults to empty dict
        is_first_module: Whether this is the first module in the network, defaults to False
    """
    
    def __init__(
        self, 
        org_module: Union[nn.Conv2d, nn.Linear],
        weight_quant_params: dict = None, 
        act_quant_params: dict = None, 
        is_first_module: bool = False
    ):
        super().__init__()
        
        # Handle default parameters
        weight_quant_params = weight_quant_params or {}
        act_quant_params = act_quant_params or {}
        
        # Setup forward function and parameters
        self._setup_forward_function(org_module)
        
        # Save original weights and biases
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        # Quantization state flags
        self.use_weight_quant = False
        self.use_act_quant = False

        # Initialize quantizers
        self.weight_quantizer = Uniform_Affine_Quantizer(**weight_quant_params)
        self.act_quantizer = self._create_act_quantizer(org_module, act_quant_params, is_first_module)

        # Training state and statistics
        self.trained = False
        self.weight_num: Optional[int] = None
        self.act_num: Optional[int] = None

    def _setup_forward_function(self, org_module: Union[nn.Conv2d, nn.Linear]) -> None:
        """Setup forward function and parameters"""
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = {
                'stride': org_module.stride,
                'padding': org_module.padding,
                'dilation': org_module.dilation,
                'groups': org_module.groups
            }
            self.fwd_func = F.conv2d
            self.type_is = "Conv2d"
        elif isinstance(org_module, nn.Linear):
            self.fwd_kwargs = {}
            self.fwd_func = F.linear
            self.type_is = "Linear"
        else:
            raise TypeError(f"Unsupported module type: {type(org_module)}")

    def _create_act_quantizer(
        self, 
        org_module: Union[nn.Conv2d, nn.Linear], 
        act_quant_params: dict, 
        is_first_module: bool
    ) -> Union[Uniform_Affine_Quantizer, nn.Identity]:
        """Create activation quantizer"""
        if isinstance(org_module, nn.Conv2d) and is_first_module:
            return nn.Identity()
        return Uniform_Affine_Quantizer(**act_quant_params)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Update statistics
        self._update_statistics(input)
        
        # Apply activation quantization
        if self.use_act_quant:
            input = self.act_quantizer(input)
        
        # Select weights and biases
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        # Execute forward pass
        return self.fwd_func(input, weight, bias, **self.fwd_kwargs)

    def _update_statistics(self, input: torch.Tensor) -> None:
        """Update statistics"""
        if self.act_num is None:
            self.act_num = input[0].numel()
        if self.weight_num is None:
            self.weight_num = self.weight.numel()

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        """Set quantization state for the module"""
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def extra_repr(self) -> str:
        """Return string representation of the quantized module"""
        if isinstance(self.act_quantizer, nn.Identity):
            return f'(wbit): {self.weight_quantizer.n_bits}\n(use_weight_quant): {self.use_weight_quant}'
        else:
            return (f'(wbit): {self.weight_quantizer.n_bits}\n'
                   f'(abit): {self.act_quantizer.n_bits}\n'
                   f'(use_weight_quant): {self.use_weight_quant}\n'
                   f'(use_act_quant): {self.use_act_quant}')


class QuantMatMul(nn.Module):
    """
    Quantized matrix multiplication class.
    
    Args:
        input_quant_params: Input quantization parameters dict, defaults to empty dict
    """
    
    def __init__(self, input_quant_params: dict = None):
        super().__init__()
        
        # Handle default parameters
        input_quant_params = input_quant_params or {}
        
        # Remove potentially existing 'log_quant' parameter
        input_quant_params = {k: v for k, v in input_quant_params.items() if k != 'log_quant'}

        # Initialize quantizers for inputs A and B
        self.quantizer_A = Uniform_Affine_Quantizer(**input_quant_params)
        self.quantizer_B = Uniform_Affine_Quantizer(**input_quant_params)

        # Quantization state flags
        self.use_weight_quant = False
        self.use_act_quant = False

        # Training state and activation count
        self.trained = False
        self.act_num: Optional[int] = None

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Forward pass for quantized matrix multiplication"""
        # Update activation count statistics
        if self.act_num is None:
            self.act_num = A[0].numel() + B[0].numel()

        # Apply activation quantization
        if self.use_act_quant:
            A = self.quantizer_A(A)
            B = self.quantizer_B(B)

        # Execute matrix multiplication
        return A @ B

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        """Set quantization state for the module"""
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def extra_repr(self) -> str:
        """Return string representation of the quantized matrix multiplication"""
        return f'(bit): {self.quantizer_A.n_bits}\n(use_act_quant): {self.use_act_quant}'

