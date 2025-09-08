from concurrent.futures import ThreadPoolExecutor
import gc
import time
from typing import Optional
import torch
import torch.nn as nn
import os


def calculate_optimal_blocks_from_model(blocks, device, config=None):
    """
    Calculate optimal blocks_to_swap based on research-backed memory analysis
    and actual training configuration including image processing specifics.
    """
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using default blocks_to_swap=20")
        return 20

    # Parse TOML configuration
    train_config = {}
    dataset_config = {}
    framework = "unknown"

    if config:
        train_config = config.get('train', {})
        dataset_config = config.get('dataset', {})
        framework = config.get('runner', {}).get('framework', 'unknown')

        # Load dataset config from specified path if it exists
        dataset_config_path = train_config.get('dataset_config')
        if dataset_config_path and not dataset_config:
            try:
                import toml
                with open(dataset_config_path, 'r') as f:
                    dataset_file = toml.load(f)
                    dataset_config = dataset_file.get('general', {})
                    if 'datasets' in dataset_file and len(dataset_file['datasets']) > 0:
                        dataset_config.update(dataset_file['datasets'][0])
            except Exception as e:
                print(f"Warning: Could not load dataset config from {dataset_config_path}: {e}")

    # Extract key configuration parameters
    optimizer_type = train_config.get('optimizer_type', 'adamw').lower()
    fp8_base = train_config.get('fp8_base', False)
    fp8_scaled = train_config.get('fp8_scaled', False)
    fp8_vl = train_config.get('fp8_vl', False)
    gradient_checkpointing = train_config.get('gradient_checkpointing', False)
    mixed_precision = train_config.get('mixed_precision', 'no').lower()
    gradient_accumulation = train_config.get('gradient_accumulation_steps', 1)

    # Image/batch specific parameters from dataset config
    batch_size = dataset_config.get('batch_size', train_config.get('batch_size', 1))
    resolution = dataset_config.get('resolution', [1024, 1024])
    if isinstance(resolution, list):
        resolution = max(resolution)  # Use the larger dimension

    # Memory state
    clean_memory_on_device(device)
    device_props = torch.cuda.get_device_properties(device)
    total_memory = device_props.total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = total_memory - allocated_memory

    print(f"GPU Memory: {free_memory / (1024 ** 3):.1f}GB free / {total_memory / (1024 ** 3):.1f}GB total")

    # Calculate actual model block sizes
    block_allocation_data = []
    total_model_memory = 0
    for i, block in enumerate(blocks):
        block_size = 0
        for param in block.parameters():
            block_size += param.numel() * param.element_size()
        for buffer in block.buffers():
            block_size += buffer.numel() * buffer.element_size()
        block_allocation_data.append((block_size, i))
        total_model_memory += block_size

    block_allocation_data.sort(reverse=True, key=lambda x: x[0])
    model_memory_gb = total_model_memory / (1024 ** 3)

    # Research-backed optimizer overhead multipliers
    optimizer_multipliers = {
        'sgd': 0.0 if not train_config.get('momentum', False) else 1.0,
        'sgd_momentum': 1.0,
        'adamw8bit': 0.25,  # 75% reduction from research
        'adamw_8bit': 0.25,
        'adamw': 2.0,  # Standard AdamW overhead
        'adam': 2.0,
        'adafactor': 0.5,
        'adam_mini': 0.3,  # Average reduction from research
        'paged_adamw8bit': 0.25,
        'paged_adamw_8bit': 0.25,
    }

    optimizer_overhead = optimizer_multipliers.get(optimizer_type, 2.0)  # Default to AdamW

    # Get actual precision bytes from config including FP8
    precision_bytes = 4  # Default fp32
    if fp8_base and fp8_scaled and fp8_vl:
        precision_bytes = 1  # FP8 - 1 byte per value
        print("FP8 full precision detected - using 1 byte per activation")
    elif fp8_base or fp8_scaled:
        precision_bytes = 1.5  # Partial FP8 - mixed 1-2 bytes
        print("Partial FP8 precision detected - using 1.5 bytes per activation")
    elif mixed_precision in ['fp16', 'bf16']:
        precision_bytes = 2
        print(f"Mixed precision {mixed_precision} - using 2 bytes per activation")
    elif mixed_precision == 'fp32':
        precision_bytes = 4
        print("FP32 precision - using 4 bytes per activation")
    else:
        precision_bytes = 4  # Default fallback

    # Precision effects on memory (research shows mixed precision can increase memory)
    precision_multiplier = 1.0
    if mixed_precision in ['fp16', 'bf16']:
        # Mixed precision: 2 bytes model + 4 bytes master copy + optimizer overhead
        precision_multiplier = 1.5  # Dual storage overhead
        print(f"Mixed precision {mixed_precision} - increased storage overhead")
    elif mixed_precision == 'fp32':
        precision_multiplier = 1.0

    # FP8 reductions (when available)
    fp8_reduction = 1.0
    if fp8_base and fp8_scaled and fp8_vl:
        fp8_reduction = 0.6  # 40% reduction with full FP8
        print("Full FP8 enabled - significant memory reduction")
    elif fp8_base or fp8_scaled:
        fp8_reduction = 0.8  # 20% reduction with partial FP8
        print("Partial FP8 enabled - moderate memory reduction")

    # Image processing memory estimation using actual block sizes
    # Calculate activation memory based on actual image dimensions and batch size
    image_channels = 3  # RGB
    activation_memory_per_image = (resolution * resolution * image_channels * precision_bytes)

    # Use actual block sizes to estimate activation memory more accurately
    # Each block processes activations, estimate based on actual block memory footprint
    avg_block_size = sum(size for size, _ in block_allocation_data) / len(block_allocation_data)
    # Activation memory scales with model complexity - use actual block sizes as proxy
    activation_scaling_factor = (avg_block_size / (1024 ** 2)) * 0.05  # 5% of avg block size in MB
    total_activation_memory = activation_memory_per_image * batch_size * len(blocks) * activation_scaling_factor

    # Framework-specific adjustments
    if framework == "qwen_image":
        # QWEN vision-language models have higher activation overhead
        total_activation_memory *= 1.25
        print("QWEN image framework detected - increased activation memory estimate")

    if gradient_checkpointing:
        total_activation_memory *= 0.3  # Gradient checkpointing reduces activation memory
        print("Gradient checkpointing enabled - reduced activation memory")

    # Gradient memory (same size as model parameters)
    gradient_memory = total_model_memory * precision_multiplier

    # Optimizer state memory
    optimizer_memory = total_model_memory * optimizer_overhead * precision_multiplier * fp8_reduction

    # Total overhead calculation
    base_overhead = (gradient_memory + optimizer_memory + total_activation_memory) / free_memory

    # Gradient accumulation reduces peak memory per step
    if gradient_accumulation > 1:
        base_overhead *= (1.0 / gradient_accumulation) * 0.8  # Partial reduction
        print(f"Gradient accumulation ({gradient_accumulation} steps) - reduced peak memory")

    # Standard safety margin
    safety_margin = 0.1  # 10% safety margin across all hardware

    # Total overhead calculation
    total_overhead = base_overhead + safety_margin

    # Memory budget calculation
    memory_budget = free_memory * (1.0 - total_overhead)

    print(f"Memory Analysis:")
    print(f"  Model memory: {model_memory_gb:.1f}GB")
    print(f"  Optimizer: {optimizer_type} (overhead: {optimizer_overhead}x)")
    print(f"  Precision bytes: {precision_bytes}")
    print(f"  Precision multiplier: {precision_multiplier:.1f}x")
    print(f"  FP8 reduction: {fp8_reduction:.1f}x")
    print(f"  Image activation memory: {total_activation_memory / (1024 ** 3):.1f}GB")
    print(f"  Total overhead: {total_overhead:.2f}")
    print(f"  Memory budget: {memory_budget / (1024 ** 3):.1f}GB")

    # Greedy allocation with calculated budget
    allocated_memory = 0
    gpu_blocks_count = 0

    for block_size, block_idx in block_allocation_data:
        if allocated_memory + block_size <= memory_budget:
            allocated_memory += block_size
            gpu_blocks_count += 1
        else:
            break

    blocks_to_swap = len(blocks) - gpu_blocks_count
    blocks_to_swap = max(1, min(blocks_to_swap, len(blocks) - 1))

    # Final safety adjustment based on empirical findings
    if optimizer_type in ['adamw8bit', 'adamw_8bit', 'paged_adamw8bit']:
        # 8-bit optimizers are more memory stable, can be slightly more aggressive
        blocks_to_swap = max(1, blocks_to_swap + 1)
    else:
        # Standard optimizers need more conservative allocation
        blocks_to_swap = max(1, blocks_to_swap + 2)

    print(f"Auto-calculated blocks_to_swap: {blocks_to_swap}")
    print(f"Keeping {len(blocks) - blocks_to_swap}/{len(blocks)} blocks on GPU")
    print(f"Estimated GPU memory usage: {allocated_memory / (1024 ** 3):.1f}GB")

    return blocks_to_swap


def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []

    # This is not working for all cases (e.g. SD3), so we need to find the corresponding modules
    # for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
    #     print(module_to_cpu.__class__, module_to_cuda.__class__)
    #     if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
    #         weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    # print(
                    #     f"Module {module_to_cuda_name} not found in CPU model or shape mismatch, so not swapping and moving to device"
                    # )
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    synchronize_device()

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    synchronize_device()


def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=True)


class Offloader:
    """
    common offloading class
    """

    def __init__(self, block_type: str, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(f"[{self.block_type}] Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter()-start_time:.2f}s")
            return bidx_to_cpu, bidx_to_cuda  # , event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter()-start_time:.2f}s")


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
            self,
            block_type: str,
            blocks: list[nn.Module],
            num_blocks: int,
            blocks_to_swap: int,
            supports_backward: bool,
            device: torch.device,
            debug: bool = False,
    ):
        # Auto-calculate if enabled
        if os.getenv('AUTO_BLOCKS_TO_SWAP') == 'true' and blocks_to_swap is not None:
            auto_calculated = calculate_optimal_blocks_from_model(blocks, device)
            print(f"Overriding blocks_to_swap: {auto_calculated} (was: {blocks_to_swap})")
            blocks_to_swap = auto_calculated

        super().__init__(block_type, num_blocks, blocks_to_swap, device, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure weights are on device

        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)  # move block to device first. this makes sure that buffers (non weights) are on the device
            weighs_to_device(b, "cpu")  # make sure weights are on cpu

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        # if backward is enabled, we do not swap blocks in forward pass more than blocks_to_swap, because it should be on GPU
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this works for forward-only offloading
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
