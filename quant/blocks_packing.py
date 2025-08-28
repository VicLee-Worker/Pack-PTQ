import torch
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple, Optional, Any, Dict

from quant.quant_module import QuantMatMul, QuantModule


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph.
    """
    pass


class DataHook:
    """Hook for capturing input/output data during forward pass."""

    def __init__(self, store_input: bool = False, store_output: bool = False, stop_forward: bool = False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch=None):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GradHook:
    """Hook for capturing gradient data during backward pass."""

    def __init__(self, store_input_grad: bool = False, store_output_grad: bool = False, stop_forward: bool = False):
        self.store_input_grad = store_input_grad
        self.store_output_grad = store_output_grad
        self.stop_forward = stop_forward

        self.input_grad = None
        self.output_grad = None

    def __call__(self, module, input_grad, output_grad=None):
        if self.store_input_grad:
            self.input_grad = input_grad
        if self.store_output_grad:
            self.output_grad = output_grad
        if self.stop_forward:
            raise StopForwardException


class ModelAnalyzer:
    """Utility class for model analysis and block extraction."""

    @staticmethod
    def get_block_type(model_name: str, q_model):
        """Get the block type based on model name."""
        block_mapping = {
            'vit': lambda m: m.model.blocks[0],
            'deit': lambda m: m.model.blocks[0],
            'swin': lambda m: m.model.layers[0].blocks[0],
            'resnet': lambda m: m.model.layer1[0],
            'mobilenetv2': lambda m: m.model.features[1],
            'mnasnet': lambda m: m.model.layers[8][0],
        }
        
        # Check for exact matches or prefix matches
        for prefix, block_getter in block_mapping.items():
            if model_name.startswith(prefix):
                return block_getter(q_model)
        
        # Special case for regnet
        if model_name.startswith('regnetx_'):
            return q_model.model.s1.b1
            
        raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def extract_model_components(q_model, block_type) -> Tuple[List, List]:
        """Extract blocks and quantized modules from the model."""
        model_blocks_list = []
        model_modules_list = []
        
        for name, module in q_model.named_modules():
            if isinstance(module, type(block_type)):
                model_blocks_list.append([name, module])
            if isinstance(module, QuantModule):
                model_modules_list.append([name, module])
                
        return model_blocks_list, model_modules_list

    @staticmethod
    def set_quant_modules_state(modules: List, weight_quant: bool, act_quant: bool):
        """Set quantization state for a list of modules."""
        for _, module in modules:
            for submodule in module.modules():
                if isinstance(submodule, (QuantModule, QuantMatMul)):
                    submodule.set_quant_state(weight_quant, act_quant)


class BatchProcessor:
    """Handles batch processing of calibration data."""
    
    def __init__(self, batch_size: int, device):
        self.batch_size = batch_size
        self.device = device
    
    def process_batches(self, data: torch.Tensor):
        """Generator for processing data in batches."""
        num_batches = int(data.size(0) / self.batch_size)
        try:
            for i in range(num_batches):
                data_range = slice(i * self.batch_size, (i + 1) * self.batch_size)
                batch_data = data[data_range].to(self.device)
                yield i, batch_data
                # Note: moving original data slice back to CPU here has no effect on GPU memory
                # because a new tensor was created for batch_data via .to(self.device).
        finally:
            torch.cuda.empty_cache()


class HessianCalculator:
    """Calculates Hessian and Omega scores for model blocks."""
    
    def __init__(self, model, batch_processor: BatchProcessor):
        self.model = model
        self.batch_processor = batch_processor
        
    def calculate_block_scores(self, blocks_list: List, cali_data: torch.Tensor, 
                             loss_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Calculate Hessian and Omega scores for all blocks."""
        hessian_scores = []
        omega_scores = []
        
        self.model.set_quant_state(False, False)
        
        for num, (name, block) in enumerate(tqdm(blocks_list, desc="Calculating Hessian scores")):
            hessian_score, omega_score = self._calculate_single_block_score(
                block, cali_data, loss_list[num]
            )
            hessian_scores.append(hessian_score)
            omega_scores.append(omega_score)
            
        return hessian_scores, omega_scores
    
    def _calculate_single_block_score(self, block, cali_data: torch.Tensor, 
                                    block_loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate Hessian and Omega scores for a single block using streaming accumulators to reduce GPU memory."""
        # Forward hook to capture block outputs
        data_hook = DataHook(store_output=True)
        data_handle = block.register_forward_hook(data_hook)

        # Accumulators on CPU to avoid GPU memory bloat
        sum_gxd = torch.tensor(0.0, dtype=torch.float64)
        sum_sq = torch.tensor(0.0, dtype=torch.float64)
        sum_abs = torch.tensor(0.0, dtype=torch.float64)
        total_elems: int = 0

        # Ensure model is in eval mode and FP by default
        self.model.set_quant_state(False, False)
        self.model.eval()

        try:
            for _, batch_data in self.batch_processor.process_batches(cali_data):
                # 1) Quantized pass for this block to get teacher logits and quantized block output
                ModelAnalyzer.set_quant_modules_state([["", block]], True, True)
                with torch.inference_mode():
                    teacher_logits = self.model(batch_data)
                    q_block_out = data_hook.output_store
                    # Detach & move quantized block output to CPU immediately
                    q_block_out_cpu = q_block_out.detach().to("cpu", dtype=torch.float32)
                # 2) Full precision pass for this block to get fp outputs and grad w.r.t block output
                ModelAnalyzer.set_quant_modules_state([["", block]], False, False)

                teacher_prob = F.softmax(teacher_logits.detach(), dim=-1)

                # Forward (with grad) in FP
                fp_pred = self.model(batch_data)
                loss = F.kl_div(
                    F.log_softmax(fp_pred, dim=-1),
                    teacher_prob,
                    reduction="batchmean"
                )

                # Get gradient w.r.t the block output directly, no param grads allocated
                fp_block_out = data_hook.output_store
                block_out_grad = torch.autograd.grad(
                    loss, fp_block_out, retain_graph=False, create_graph=False, allow_unused=True
                )[0]

                # Move to CPU
                fp_block_out_cpu = fp_block_out.detach().to("cpu", dtype=torch.float32)
                block_out_grad_cpu = (block_out_grad.detach().to("cpu", dtype=torch.float32)
                                      if block_out_grad is not None else torch.zeros_like(fp_block_out_cpu))

                # Compute diffs on CPU and update accumulators
                diff_cpu = fp_block_out_cpu - q_block_out_cpu
                sum_gxd += torch.sum(block_out_grad_cpu * diff_cpu, dtype=torch.float64)
                sum_sq += 0.5 * torch.sum(diff_cpu * diff_cpu, dtype=torch.float64)
                sum_abs += torch.sum(torch.abs(diff_cpu), dtype=torch.float64)
                total_elems += diff_cpu.numel()

                # Cleanup per-batch GPU tensors
                del teacher_logits, teacher_prob, fp_pred, fp_block_out, block_out_grad, q_block_out
                torch.cuda.empty_cache()

            # Compute means and final scores on CPU
            if total_elems == 0:
                # Fallback to zeros to avoid division by zero
                first_term = torch.tensor(0.0, dtype=torch.float64)
                second_term = torch.tensor(1e-12, dtype=torch.float64)
                mean_abs = torch.tensor(0.0, dtype=torch.float64)
            else:
                first_term = sum_gxd / total_elems
                second_term = sum_sq / total_elems
                mean_abs = sum_abs / total_elems

            # Ensure block_loss is on CPU float64 for stable computation
            block_loss_cpu = block_loss.detach().to("cpu", dtype=torch.float64)
            # Avoid divide-by-zero
            eps = torch.tensor(1e-12, dtype=torch.float64)
            denom = torch.maximum(second_term, eps)
            hessian_score = (block_loss_cpu - first_term) / denom
            omega_score = hessian_score * torch.abs(mean_abs)

            # Return 0-dim tensors (CPU) to keep behavior consistent with callers
            return hessian_score.to(dtype=torch.float32), omega_score.to(dtype=torch.float32)
        finally:
            data_handle.remove()


def get_block_type(name, q_model):
    """Legacy function - kept for backward compatibility."""
    return ModelAnalyzer.get_block_type(name, q_model)


def split_modules(all_modules: List, blocks: List) -> Tuple[List, List, List]:
    """
    Splits the modules into before and after blocks, and inserts modules between blocks.

    Args:
        all_modules: A list of all modules.
        blocks: A list of blocks to split the modules around.

    Returns:
        A tuple containing before_blocks_modules, after_blocks_modules, and new_blocks.
    """
    before_blocks_modules = []
    after_blocks_modules = []
    block_names = [block[0] for block in blocks]
    in_block = False

    # Separate modules into before and after blocks
    for name, module in all_modules:
        if any(block_name in name for block_name in block_names):
            in_block = True
            after_blocks_modules = []
            continue
        if in_block:
            after_blocks_modules.append([name, module])
        else:
            before_blocks_modules.append([name, module])

    # Insert modules between blocks
    new_blocks = []
    for i, (block_name, block_module) in enumerate(blocks):
        new_blocks.append([block_name, block_module])
        if i < len(blocks) - 1:
            block_flag = False
            next_block_flag = False
            next_block_name = blocks[i + 1][0]
            for name, module in all_modules:
                if block_name in name:
                    block_flag = True
                    continue
                if next_block_name in name:
                    next_block_flag = True
                if block_flag and not next_block_flag:
                    new_blocks.append([name, module])

    return before_blocks_modules, after_blocks_modules, new_blocks


class AdaptiveBlocksPacker:
    """Main class for adaptive blocks packing functionality."""
    
    def __init__(self, args, q_model, device):
        self.args = args
        self.q_model = q_model
        self.device = device
        self.batch_processor = BatchProcessor(args.batch_size, device)
        self.hessian_calculator = HessianCalculator(q_model, self.batch_processor)
    
    def run_packing(self, cali_data: torch.Tensor, **kwargs) -> Tuple[List, List, List, List]:
        """
        Main method to perform adaptive blocks packing.
        
        Returns:
            Tuple of (blocks_packing_list, blocks_packing_list_score, 
                     before_blocks_list, after_blocks_list)
        """
        logger.info("Beginning Adaptive blocks packing:")
        
        # Extract model components
        block_type = ModelAnalyzer.get_block_type(self.args.model_name, self.q_model)
        model_blocks_list, model_modules_list = ModelAnalyzer.extract_model_components(
            self.q_model, block_type
        )
        
        # Split modules
        before_blocks_list, after_blocks_list, model_blocks_list = split_modules(
            model_modules_list, model_blocks_list
        )
        
        # Log model blocks
        block_names = [name for name, _ in model_blocks_list]
        logger.info(f"Model blocks list: {block_names}")
        
        # Calculate loss errors
        loss_list = self._calculate_loss_errors(model_blocks_list, cali_data)
        
        # Calculate Hessian and Omega scores
        hessian_scores, omega_scores = self.hessian_calculator.calculate_block_scores(
            model_blocks_list, cali_data, loss_list
        )
        
        # Create blocks packing list
        blocks_packing_list, blocks_packing_list_score = self._create_packing_list(
            model_blocks_list, hessian_scores, omega_scores
        )
        
        # Log results
        self._log_results(blocks_packing_list)
        
        return blocks_packing_list, blocks_packing_list_score, before_blocks_list, after_blocks_list
    
    def _calculate_loss_errors(self, model_blocks_list: List, cali_data: torch.Tensor) -> List[torch.Tensor]:
        """Calculate loss errors for each block."""
        # logger.info("Recording loss errors...")
        
        pred = [[] for _ in range(len(model_blocks_list))]
        fp_pred = []
        
        self.q_model.set_quant_state(False, False)

        total_batches = int(cali_data.size(0) // self.batch_processor.batch_size)
        
        with torch.no_grad():
            for i, batch_data in tqdm(self.batch_processor.process_batches(cali_data), total=total_batches, desc="Calculating loss errors"):
                # Calculate predictions for each block with quantization
                for num, (name, block) in enumerate(model_blocks_list):
                    ModelAnalyzer.set_quant_modules_state([["", block]], True, True)
                    pred[num].append(self.q_model(batch_data).detach().cpu())
                    ModelAnalyzer.set_quant_modules_state([["", block]], False, False)
                
                # Full precision prediction
                self.q_model.set_quant_state(False, False)
                fp_pred.append(self.q_model(batch_data).detach().cpu())
        
        # Concatenate and calculate losses (on CPU)
        pred = [torch.cat(each_pred, dim=0) for each_pred in pred]
        pred_softmax = [F.softmax(x, dim=-1) for x in pred]
        fp_pred = torch.cat(fp_pred, dim=0)
        
        loss_list = []
        for each_pred in pred_softmax:
            loss = F.kl_div(
                F.log_softmax(fp_pred, dim=-1), 
                each_pred, 
                reduction="batchmean"
            )
            loss_list.append(loss)
        
        return loss_list
    
    def _create_packing_list(self, model_blocks_list: List, hessian_scores: List, 
                            omega_scores: List) -> Tuple[List, List]:
        """Create the final blocks packing list based on scores."""
        sorted_hessian_scores = sorted(enumerate(hessian_scores), key=lambda x: x[1])
        
        blocks_packing_list = []
        blocks_packing_list_score = []
        back = len(model_blocks_list)
        
        for index, _ in sorted_hessian_scores:
            front = index
            if front < back:
                pack = model_blocks_list[front:back]
                blocks_packing_list.insert(0, pack)
                
                # Calculate average score for this pack
                avg_score = sum(omega_scores[front:back]) / (back - front) if back > front else 0
                blocks_packing_list_score.insert(0, avg_score)
                
                back = index
        
        return blocks_packing_list, blocks_packing_list_score
    
    def _log_results(self, blocks_packing_list: List):
        """Log the final blocks packing results."""
        result = {}
        for num, pack in enumerate(blocks_packing_list):
            pack_name = f'pack {num + 1}'
            block_names = [name for name, _ in pack]
            result[pack_name] = block_names
        logger.info(f"Blocks packing list: {result}")


def adaptive_blocks_packing(args, q_model, cali_data, training_kwargs) -> Tuple[List, List, List, List]:
    """
    Main function for adaptive blocks packing - refactored for better maintainability.
    
    Args:
        args: Command-line arguments.
        q_model: The quantized model.
        cali_data: Calibration data.
        training_kwargs: Additional keyword arguments (renamed from kwargs to avoid conflicts).

    Returns:
        Tuple of (blocks_packing_list, blocks_packing_list_score, 
                 before_blocks_list, after_blocks_list)
    """
    device = next(q_model.parameters()).device
    packer = AdaptiveBlocksPacker(args, q_model, device)
    
    # Remove cali_data from training_kwargs if it exists to avoid parameter conflicts
    clean_kwargs = {k: v for k, v in training_kwargs.items() if k != 'cali_data'}
    
    return packer.run_packing(cali_data, **clean_kwargs)