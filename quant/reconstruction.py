from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from quant.quant_module import AdaRoundQuantizer, QuantMatMul, QuantModule
import os
from loguru import logger
from typing import List, Optional, Tuple, Any, Dict
from utils.accuracy import validate_model

# Import StopForwardException from blocks_packing
from quant.blocks_packing import StopForwardException


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block.
    """
    def __init__(self, store_input: bool = False, store_output: bool = False, 
                 stop_forward: bool = False, modify_data: bool = False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.modify_data = modify_data

        self.input_store = None
        self.output_store = None
        self.modified_input_store = None

    def __call__(self, module, input_batch, output_batch=None):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.modify_data:
            return self.modified_input_store
        if self.stop_forward:
            raise StopForwardException


def _find_sequential_parent_and_indices(model: nn.Module,
                                        first_name: str,
                                        last_name: str,
                                        first_module: nn.Module,
                                        last_module: nn.Module):
    """
    Try to find a common parent that is Sequential/ModuleList and indices of first/last under that parent.
    This function recursively searches up the module hierarchy.
    Returns (parent, start_idx, end_idx) or (None, None, None) if not possible.
    """
    named_mods = dict(model.named_modules())

    def get_ancestors(name):
        parts = name.split('.')
        ancestors = []
        for i in range(len(parts) - 1, 0, -1):
            parent_name = '.'.join(parts[:i])
            parent_module = named_mods.get(parent_name)
            child_name = parts[i]
            if parent_module:
                ancestors.append((parent_name, parent_module, child_name))
        ancestors.append(('', model, name.split('.')[0])) # Add root model
        return ancestors

    ancestors1 = get_ancestors(first_name)
    ancestors2 = get_ancestors(last_name)

    # Find common ancestor
    for p1_name, p1_mod, c1_name in ancestors1:
        for p2_name, p2_mod, c2_name in ancestors2:
            if p1_name == p2_name:
                parent = p1_mod
                if isinstance(parent, (nn.Sequential, nn.ModuleList)):
                    i1, i2 = None, None
                    children = list(parent.children())
                    for i, child in enumerate(children):
                        if child is named_mods.get(f"{p1_name}.{c1_name}" if p1_name else c1_name):
                            i1 = i
                        if child is named_mods.get(f"{p2_name}.{c2_name}" if p2_name else c2_name):
                            i2 = i
                    
                    if i1 is not None and i2 is not None and i1 <= i2:
                        return parent, i1, i2
    
    return None, None, None


def get_module_data(q_model: nn.Module, module: nn.Module, cali_data: torch.Tensor,
                    batch_size: int = 32, is_quant_data: bool = False, data_type: str = 'input') -> torch.Tensor:
    """
    Get the input or output data for a specific module.

    Args:
        q_model (nn.Module): The quantized model.
        module (nn.Module): The module to get data for.
        cali_data (torch.Tensor): The calibration data.
        batch_size (int, optional): The batch size. Defaults to 32.
        is_quant_data (bool, optional): Whether to use quantized data. Defaults to False.
        data_type (str, optional): Type of data to get ('input' or 'output'). Defaults to 'input'.

    Returns:
        torch.Tensor: The input or output data for the module.
    """
    q_model.set_quant_state(is_quant_data, is_quant_data)

    if data_type == 'input':
        hook = DataSaverHook(store_input=True, stop_forward=True)
        handle = module.register_forward_pre_hook(hook)
    else:
        hook = DataSaverHook(store_output=True, stop_forward=True)
        handle = module.register_forward_hook(hook)

    device = next(q_model.parameters()).device
    data_store = []

    with torch.no_grad():
        for i in range(int(cali_data.size(0) / batch_size)):
            try:
                _ = q_model(cali_data[i * batch_size:(i + 1) * batch_size].to(device))
            except StopForwardException:
                pass
            
            if data_type == 'input':
                stored_data = hook.input_store[0]
            else:
                stored_data = hook.output_store
            
            data_store.append(stored_data.detach().cpu())

    handle.remove()
    
    all_data = torch.cat(data_store)
    torch.cuda.empty_cache()
    return all_data.to(device)


class ReconstructionConfig:
    """Configuration for module reconstruction."""
    
    def __init__(self, batch_size: int = 16, iters: int = 20000, weight: float = 0.01, 
                 opt_mode: str = 'mse', b_range: Tuple[int, int] = (20, 2), warmup: float = 0.0,
                 p: float = 2.0, lr: float = 4e-5, input_prob: float = 1.0, 
                 lamb_r: float = 0.2, T: float = 7.0, bn_lr: float = 1e-3, 
                 lamb_c: float = 0.0, cali_data: Optional[torch.Tensor] = None, **kwargs):
        self.batch_size = batch_size
        self.iters = iters
        self.weight = weight
        self.opt_mode = opt_mode
        self.b_range = b_range
        self.warmup = warmup
        self.p = p
        self.lr = lr
        self.input_prob = input_prob
        self.lamb_r = lamb_r
        self.T = T
        self.bn_lr = bn_lr
        self.lamb_c = lamb_c
        self.loss_calc_start = int(0.95 * iters)  # Start calculating final loss
        
        # Store any additional kwargs for future extensibility
        for key, value in kwargs.items():
            setattr(self, key, value)


class PackExecutor:
    """Handles execution for packed modules."""
    
    def __init__(self, q_model: nn.Module, modules_to_reconstruct: list, config: ReconstructionConfig, logger, use_fast_path: bool = True):
        self.q_model = q_model
        self.modules = modules_to_reconstruct
        self.config = config
        self.logger = logger
        self.device = next(q_model.parameters()).device
        
        self.use_fast_path = use_fast_path
        self.fast_runner = None
        self.first_block_hook = None
        self.last_block_hook = None
        self.hook_handles = []
        
    def setup_execution_path(self, cali_data: torch.Tensor):
        """Setup execution path for packed modules."""
        first_block = self.modules[0][1]
        last_block = self.modules[-1][1]
        
        # Try to find fast execution path
        if self.use_fast_path:
            fast_parent, start_idx, end_idx = _find_sequential_parent_and_indices(
                self.q_model, self.modules[0][0], self.modules[-1][0], first_block, last_block
            )
        else:
            fast_parent, start_idx, end_idx = None, None, None
        
        if fast_parent is not None:
            self._setup_fast_path(fast_parent, start_idx, end_idx, cali_data)
        else:
            self._setup_hook_path(first_block, last_block, cali_data)
    
    def _setup_fast_path(self, parent, start_idx: int, end_idx: int, cali_data: torch.Tensor):
        """Setup fast execution using Sequential/ModuleList parent."""
        children = list(parent.children())
        
        def _fast_run_pack(x):
            out = x
            for i in range(start_idx, end_idx + 1):
                out = children[i](out)
            return out
        
        self.fast_runner = _fast_run_pack
        fast_runner_start = children[start_idx]
        first_block = self.modules[0][1]
        last_block = self.modules[-1][1]
        
        # Prepare data
        self.fast_runner_input = torch.zeros_like(get_module_data(self.q_model, fast_runner_start, cali_data, 
                                                                  self.config.batch_size, is_quant_data=True, data_type='input'))
        self.input_data = get_module_data(self.q_model, first_block, cali_data, 
                                        self.config.batch_size, is_quant_data=True, data_type='input')
        self.output_data = get_module_data(self.q_model, last_block, cali_data, 
                                         self.config.batch_size, is_quant_data=False, data_type='output')
        self.fp_input_data = get_module_data(self.q_model, first_block, cali_data, 
                                           self.config.batch_size, is_quant_data=False, data_type='input')
        # Setup hooks
        self.first_block_hook = DataSaverHook(store_input=False, store_output=False, 
                                            stop_forward=False, modify_data=True)
        self.last_block_hook = DataSaverHook(store_input=False, store_output=True, stop_forward=True)
        
        handle1 = first_block.register_forward_pre_hook(self.first_block_hook)
        handle2 = last_block.register_forward_hook(self.last_block_hook)
        self.hook_handles = [handle1, handle2]
        
        self.logger.info(f"Fast pack path enabled: parent={type(parent).__name__}, slice=[{start_idx}, {end_idx}]")
    
    def _setup_hook_path(self, first_block, last_block, cali_data: torch.Tensor):
        """Setup hook-based execution for packed modules."""
        self.logger.warning("Fast pack path unavailable. Falling back to hook-based execution.")
        
        # Prepare data
        self.input_data = get_module_data(self.q_model, first_block, cali_data, 
                                        self.config.batch_size, is_quant_data=True, data_type='input')
        self.output_data = get_module_data(self.q_model, last_block, cali_data, 
                                         self.config.batch_size, is_quant_data=False, data_type='output')
        self.fp_input_data = get_module_data(self.q_model, first_block, cali_data, 
                                           self.config.batch_size, is_quant_data=False, data_type='input')
        
        # Setup hooks
        self.first_block_hook = DataSaverHook(store_input=False, store_output=False, 
                                            stop_forward=False, modify_data=True)
        self.last_block_hook = DataSaverHook(store_input=False, store_output=True, stop_forward=True)
        
        handle1 = first_block.register_forward_pre_hook(self.first_block_hook)
        handle2 = last_block.register_forward_hook(self.last_block_hook)
        self.hook_handles = [handle1, handle2]

    def execute_forward(self, mixed_input: torch.Tensor, cali_data: torch.Tensor, batch_indices) -> torch.Tensor:
        """Execute forward pass for packed modules."""
        self.first_block_hook.modified_input_store = mixed_input
        if self.fast_runner is not None:
            self.fast_runner(self.fast_runner_input[batch_indices].to(self.device))
        else:
            _ = self.q_model(cali_data[0].unsqueeze(0).to(self.device))
    
    def cleanup(self):
        """Clean up hooks."""
        for handle in self.hook_handles:
            handle.remove()


class ModuleReconstructor:
    """Main class for module reconstruction."""
    
    def __init__(self, q_model: nn.Module, config: ReconstructionConfig, logger):
        self.q_model = q_model
        self.config = config
        self.logger = logger
        self.device = next(q_model.parameters()).device
        
    def _prepare_data_for_single_module(self, module, cali_data: torch.Tensor):
        """Prepare input/output data for single module reconstruction."""
        input_data = get_module_data(self.q_model, module, cali_data, 
                                   self.config.batch_size, is_quant_data=True, data_type='input')
        output_data = get_module_data(self.q_model, module, cali_data, 
                                    self.config.batch_size, is_quant_data=False, data_type='output')
        return input_data, output_data
    
    def _setup_quantization_parameters(self, modules_to_reconstruct: list):
        """Setup quantization parameters for reconstruction."""
        # Freeze all model parameters first
        for param in self.q_model.parameters():
            param.requires_grad = False
        
        round_mode = 'learned_hard_sigmoid'
        w_params, a_params = [], []
        
        for module_name, module in modules_to_reconstruct:
            for submodule in module.modules():
                if isinstance(submodule, QuantModule):
                    self._setup_quant_module(submodule, round_mode, w_params, a_params)
                elif isinstance(submodule, QuantMatMul):
                    self._setup_quant_matmul(submodule, a_params)
        
        return w_params, a_params
    
    def _setup_quant_module(self, module: QuantModule, round_mode: str, w_params: list, a_params: list):
        """Setup parameters for QuantModule."""
        # Setup weight quantizer
        if not isinstance(module.weight_quantizer, AdaRoundQuantizer):
            module.weight_quantizer = AdaRoundQuantizer(
                uaq=module.weight_quantizer,
                round_mode=round_mode,
                weight_tensor=module.org_weight.data
            )
            module.weight_quantizer.soft_targets = True
        
        module.weight_quantizer.alpha.requires_grad = True
        w_params.append(module.weight_quantizer.alpha)
        
        # Setup activation quantizer if needed
        if (not isinstance(module.act_quantizer, torch.nn.Identity) and 
            module.act_quantizer.delta is not None):
            module.act_quantizer.delta.requires_grad = True
            a_params.append(module.act_quantizer.delta)
    
    def _setup_quant_matmul(self, module: QuantMatMul, a_params: list):
        """Setup parameters for QuantMatMul."""
        if module.quantizer_A.delta is not None:
            module.quantizer_A.delta.requires_grad = True
            a_params.append(module.quantizer_A.delta)
        if module.quantizer_B.delta is not None:
            module.quantizer_B.delta.requires_grad = True
            a_params.append(module.quantizer_B.delta)
    
    def _create_optimizers(self, w_params: list, a_params: list):
        """Create optimizers for weight and activation parameters."""
        w_opt = torch.optim.Adam(w_params, lr=3e-3) if w_params else None
        a_opt = torch.optim.Adam(a_params, lr=self.config.lr) if a_params else None
        a_scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=self.config.iters, eta_min=0.) 
                      if a_opt else None)
        return w_opt, a_opt, a_scheduler

    def _prepare_batch_data(self, is_pack: bool, input_data: torch.Tensor, 
                          output_data: torch.Tensor, fp_input_data: torch.Tensor, 
                          batch_indices: torch.Tensor):
        """Prepare batch data for training."""
        if is_pack:
            q_inp = input_data[batch_indices].to(self.device)
            fp_inp = fp_input_data[batch_indices].to(self.device)
            target_out = output_data[batch_indices].to(self.device)
            
            if self.config.input_prob < 1.0:
                mixed_input = torch.where(
                    torch.rand_like(q_inp) < self.config.input_prob, q_inp, fp_inp
                )
            else:
                mixed_input = q_inp
        else:
            mixed_input = input_data[batch_indices].to(self.device)
            target_out = output_data[batch_indices].to(self.device)
        
        return mixed_input, target_out

    def _training_step(self, mixed_input: torch.Tensor, target_out: torch.Tensor, 
                       is_pack: bool, pack_executor: Optional[PackExecutor], single_module: Optional[nn.Module],
                       cali_data: torch.Tensor, w_opt, a_opt, a_scheduler, loss_func,
                       batch_indices) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Execute a single training step."""
        # Zero gradients
        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()
        
        # Forward pass
        try:
            if is_pack:
                _ = pack_executor.execute_forward(mixed_input, cali_data, batch_indices)
            else:
                mixed_output = single_module(mixed_input)
        except StopForwardException:
            if is_pack:
                mixed_output = pack_executor.last_block_hook.output_store
            else:
                return None, None
        except Exception as e:
            self.logger.error(f"Forward pass error: {e}")
            raise
        
        # Compute loss
        total_loss, rec_loss = loss_func(mixed_output, target_out)
        
        # Backward pass
        total_loss.backward(retain_graph=True)
        
        # Optimize
        if w_opt:
            w_opt.step()
        if a_opt:
            a_opt.step()
        if a_scheduler:
            a_scheduler.step()
        
        return total_loss, rec_loss


def Reconstruct_Modules(q_model: nn.Module, modules_to_reconstruct: list, logger, is_pack: bool, 
                        cali_data: torch.Tensor, **kwargs) -> float:
    """
    Reconstruct modules using adaptive rounding quantization.
    
    Args:
        q_model (nn.Module): The quantized model
        modules_to_reconstruct (list): List of (name, module) pairs to reconstruct
        logger: Logger instance
        is_pack (bool): Whether reconstructing a pack of modules or single module
        cali_data (torch.Tensor): Calibration data
        **kwargs: Additional configuration parameters
        
    Returns:
        float: Final reconstruction loss
    """
    config = ReconstructionConfig(**kwargs)
    reconstructor = ModuleReconstructor(q_model, config, logger)
    
    # Setup quantization parameters
    w_params, a_params = reconstructor._setup_quantization_parameters(modules_to_reconstruct)
    w_opt, a_opt, a_scheduler = reconstructor._create_optimizers(w_params, a_params)
    
    # Prepare data and execution
    if is_pack:
        pack_executor = PackExecutor(q_model, modules_to_reconstruct, config, logger)
        # pack_executor = PackExecutor(q_model, modules_to_reconstruct, config, logger, use_fast_path=False)
        pack_executor.setup_execution_path(cali_data)
        input_data = pack_executor.input_data
        output_data = pack_executor.output_data
        fp_input_data = getattr(pack_executor, 'fp_input_data', None)
        single_module = None
    else:
        pack_executor = None
        single_module = modules_to_reconstruct[0][1]
        input_data, output_data = reconstructor._prepare_data_for_single_module(single_module, cali_data)
        fp_input_data = None
    
    # Setup loss function
    loss_func = LossFunction(
        logger, modules_to_reconstruct, round_loss='relaxation', weight=config.weight,
        max_count=config.iters, rec_loss=config.opt_mode, b_range=config.b_range,
        decay_start=0, warmup=config.warmup, p=config.p, lam=config.lamb_r, T=config.T
    )
    
    # Training setup
    q_model.set_quant_state(True, True)
    data_size = input_data.size(0)
    blocks_loss = 0.0
    
    # logger.info(f"Starting reconstruction: {'pack' if is_pack else 'single module'}, "
    #            f"iterations={config.iters}, batch_size={config.batch_size}")
    
    # Training loop
    for iteration in range(config.iters):
        # Prepare batch
        batch_indices = torch.randint(0, data_size, (config.batch_size,))
        mixed_input, target_out = reconstructor._prepare_batch_data(
            is_pack, input_data, output_data, fp_input_data, batch_indices
        )
        
        # Training step
        total_loss, rec_loss = reconstructor._training_step(
            mixed_input, target_out, is_pack, pack_executor, single_module,
            cali_data, w_opt, a_opt, a_scheduler, loss_func, batch_indices
        )
        
        if total_loss is None:
            continue
        
        # Update running loss (only in final 5% of iterations)
        if iteration >= config.loss_calc_start:
            current_loss = total_loss.detach().cpu().item()
            if iteration == config.loss_calc_start:
                blocks_loss = current_loss
            else:
                # Running average
                weight = 1.0 / (iteration - config.loss_calc_start + 1)
                blocks_loss = blocks_loss * (1 - weight) + current_loss * weight
        
        # Memory cleanup
        if (iteration + 1) % 200 == 0:
            torch.cuda.empty_cache()
    
    # Cleanup
    if is_pack and pack_executor:
        pack_executor.cleanup()
    
    # logger.info(f"Reconstruction completed. Final loss: {blocks_loss:.6f}")
    return blocks_loss


def lp_loss(pred: torch.Tensor, tgt: torch.Tensor, p: float = 2.0, reduction: str = 'none') -> torch.Tensor:
    """
    Loss function measured in L_p Norm.

    Args:
        pred (torch.Tensor): The predicted tensor.
        tgt (torch.Tensor): The target tensor.
        p (float, optional): The order of the norm. Defaults to 2.0.
        reduction (str, optional): The reduction method. Defaults to 'none'.

    Returns:
        torch.Tensor: The computed loss.
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class LossFunction:
    def __init__(self,
                 logger,
                 modules: List = [],
                 round_loss: str = 'relaxation',
                 weight: float = 1.0,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: Tuple[int, int] = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.0,
                 lam: float = 1.0,
                 T: float = 7.0):
        """
        Initialize the LossFunction class.

        Args:
            logger: Logger for logging the process.
            modules (list, optional): List of modules to apply the loss function to. Defaults to [].
            round_loss (str, optional): The type of rounding loss. Defaults to 'relaxation'.
            weight (float, optional): The weight for the rounding loss. Defaults to 1.0.
            rec_loss (str, optional): The type of reconstruction loss. Defaults to 'mse'.
            max_count (int, optional): The maximum number of iterations. Defaults to 2000.
            b_range (tuple, optional): The range for the temperature schedule. Defaults to (10, 2).
            decay_start (float, optional): The start point for temperature decay. Defaults to 0.0.
            warmup (float, optional): The warmup period. Defaults to 0.0.
            p (float, optional): The order of the norm for the loss function. Defaults to 2.0.
            lam (float, optional): The regularization hyperparameter. Defaults to 1.0.
            T (float, optional): The temperature coefficient. Defaults to 7.0.
        """
        self.logger = logger
        self.modules = modules
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lam = lam
        self.T = T

        self.temp_decay = LinearTempDecay(
            max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0],
            end_b=b_range[1]
        )
        self.count = 0
        self.pd_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.loss_save = []

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor, output: torch.Tensor = None, output_fp: torch.Tensor = None) -> tuple:
        """
        Compute the total loss for adaptive rounding.

        Args:
            pred (torch.Tensor): Output from the quantized model.
            tgt (torch.Tensor): Output from the floating-point (FP) model.
            output (torch.Tensor, optional): Prediction from the quantized model. Defaults to None.
            output_fp (torch.Tensor, optional): Prediction from the FP model. Defaults to None.

        Returns:
            tuple: Total loss and reconstruction loss.
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            raise ValueError(f'Not supported reconstruction loss function: {self.rec_loss}')

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for item in self.modules:
                for module in item[1].modules():
                    if isinstance(module, QuantModule):
                        round_vals = module.weight_quantizer.get_soft_targets()
                        round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        self.loss_save.append(rec_loss.detach().cpu().float())
        if self.count % 1000 == 0:
            self.logger.info(
                f'Loss: {float(total_loss):10.3f} ( '
                f'Rec: {float(rec_loss):8.3f} | '
                f'Round: {float(round_loss):10.3f} ) | '
                f'b: {b:7.2f} | Count: {self.count}'
            )
        return total_loss, rec_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        """
        Initialize the LinearTempDecay class.

        Args:
            t_max (int): The maximum number of time steps.
            rel_start_decay (float, optional): The relative start point for decay. Defaults to 0.2.
            start_b (int, optional): The starting temperature. Defaults to 10.
            end_b (int, optional): The ending temperature. Defaults to 2.
        """
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t: int) -> float:
        """
        Linear decay scheduler for temperature b.

        Args:
            t (int): The current time step.

        Returns:
            float: The scheduled temperature.
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

class ReconstructionManager:
    """Manages the quantization reconstruction workflow."""
    
    def __init__(self, model, test_loader, device, logger):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.logger = logger
        
    def run_adaptive_packing(self, args, calibration_data, training_kwargs) -> Tuple[List, List, List, List]:
        """Run adaptive blocks packing analysis."""
        from quant.blocks_packing import adaptive_blocks_packing
        
        logger.info("Starting adaptive blocks packing...")
        
        packs, pack_scores, before_modules, after_modules = adaptive_blocks_packing(
            args, self.model, calibration_data, training_kwargs
        )
        
        logger.info(f"Adaptive packing completed: {len(packs)} packs identified")
        return packs, pack_scores, before_modules, after_modules
    
    def apply_mixed_precision(self, args, weight_quant_params, calibration_data, 
                            training_kwargs, packs, pack_scores, 
                            before_modules, after_modules):
        """Apply mixed precision quantization if enabled."""
        if not args.enable_mixed_precision:
            return
            
        from quant.mixed_precision import mixed_precision_mode
        
        logger.info("Applying mixed precision mode...")
        mixed_precision_mode(
            args, weight_quant_params, self.model, calibration_data, 
            training_kwargs, packs, pack_scores, before_modules, after_modules
        )
        logger.info("Mixed precision mode applied")
    
    def _reconstruct_pack_or_module(self, modules_to_reconstruct, is_pack, training_kwargs):
        """Helper to run reconstruction on a single module or a pack."""
        Reconstruct_Modules(self.model, modules_to_reconstruct, self.logger, is_pack=is_pack, **training_kwargs)

    def reconstruct_modules(self, packs, before_modules, after_modules, training_kwargs):
        """Perform module reconstruction in the correct order."""
        self.model.zero_grad()
        
        # Get all quantized modules (as name-module pairs)
        quant_modules_dict = {name: module for name, module in self.model.named_modules() if isinstance(module, QuantModule)}

        # Reconstruct before-packing modules
        self.logger.info("Reconstructing pre-pack modules...")
        for module_name, module in before_modules:
            self.logger.info("The module is: {}".format(module_name))
            self._reconstruct_pack_or_module([[module_name, module]], is_pack=False, training_kwargs=training_kwargs)

        # Reconstruct packs
        self.logger.info("Reconstructing packs...")
        for i, pack in enumerate(packs):
            self.logger.info("The pack {}/{} includes {} blocks, containing: {}".format(i+1, len(packs),len(pack), [item[0] for item in pack]))
            self._reconstruct_pack_or_module(pack, is_pack=True, training_kwargs=training_kwargs)
        
        # Reconstruct after-packing modules
        self.logger.info("Reconstructing post-pack modules...")
        for module_name, module in after_modules:
            self.logger.info("The module is: {}".format(module_name))
            self._reconstruct_pack_or_module([[module_name, module]], is_pack=False, training_kwargs=training_kwargs)
            
    
    def finalize_quantization(self, weight_bits: int, activation_bits: int) -> float:
        """Finalize quantization and evaluate accuracy."""
        self.model.set_quant_state(True, True)
        
        accuracy = validate_model(self.test_loader, self.model)
        logger.info(f'Final quantization (W{weight_bits}A{activation_bits}) accuracy: {accuracy:.4f}')
        
        return accuracy
    
    def save_final_model(self, save_path: str, model_info: dict = None):
        """Save the final quantized model state dict."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        
        # Add model info if provided
        if model_info:
            checkpoint.update(model_info)
            
        torch.save(checkpoint, save_path)
        logger.info(f"Final model state dict saved to {save_path}")


def run_full_reconstruction_pipeline(
    model, 
    args, 
    calibration_data, 
    test_loader,
    weight_quant_params: Dict[str, Any],
    training_kwargs: Dict[str, Any],
    save_path: str
) -> float:
    """
    Run the complete reconstruction pipeline.
    
    Args:
        model: Quantized model to reconstruct
        args: Command line arguments
        calibration_data: Calibration dataset
        test_loader: Test data loader
        weight_quant_params: Weight quantization parameters
        training_kwargs: Training configuration
        save_path: Path to save final model
        
    Returns:
        Final model accuracy
    """
    device = next(model.parameters()).device
    
    # Initialize reconstruction manager
    recon_manager = ReconstructionManager(model, test_loader, device, logger)
    
    # Step 1: Adaptive blocks packing
    packs, pack_scores, before_modules, after_modules = recon_manager.run_adaptive_packing(
        args, calibration_data, training_kwargs
    )
    
    # Step 2: Mixed precision (if enabled)
    recon_manager.apply_mixed_precision(
        args, weight_quant_params, calibration_data, training_kwargs,
        packs, pack_scores, before_modules, after_modules
    )
    
    # Step 3: Module reconstruction
    recon_manager.reconstruct_modules(
        packs, before_modules, after_modules, training_kwargs
    )
    
    # Step 4: Finalize and evaluate
    final_accuracy = recon_manager.finalize_quantization(
        args.weight_bits, args.activation_bits
    )
    
    # Step 5: Save final model
    model_info = {
        'model_name': getattr(args, 'model_name', 'unknown'),
        'weight_bits': args.weight_bits,
        'activation_bits': args.activation_bits,
        'final_accuracy': final_accuracy,
        'training_completed': True
    }
    recon_manager.save_final_model(save_path, model_info)
    
    return final_accuracy