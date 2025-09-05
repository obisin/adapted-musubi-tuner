#!/usr/bin/env python3
"""
Musubi Tuner Training Wrapper
Usage: python train.py path/to/config.toml
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import subprocess
from pathlib import Path
import gc
import torch

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def calculate_optimal_blocks_from_model(blocks, device, config):
    """Calculate optimal blocks_to_swap using actual model blocks"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using default blocks_to_swap=20")
        return 20

    # Check if VAE and text encoder are optimized/offloaded
    train_config = config.get('train', {})
    cache_latents_config = config.get('cache_latents', {})
    cache_te_config = config.get('cache_text_encoder', {})

    vae_on_cpu = train_config.get('vae_cache_cpu', False) or cache_latents_config.get('device') == 'cpu'
    text_encoder_optimized = train_config.get('fp8_t5', False) or cache_te_config.get('device') == 'cpu'

    # Cleanup and get memory state
    gc.collect()
    torch.cuda.empty_cache()

    device_props = torch.cuda.get_device_properties(device)
    total_memory = device_props.total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = total_memory - allocated_memory

    print(f"GPU Memory: {free_memory / (1024 ** 3):.1f}GB free / {total_memory / (1024 ** 3):.1f}GB total")

    # Calculate block sizes
    block_allocation_data = []
    for i, block in enumerate(blocks):
        block_size = 0
        for param in block.parameters():
            block_size += param.numel() * param.element_size()
        for buffer in block.buffers():
            block_size += buffer.numel() * buffer.element_size()
        block_allocation_data.append((block_size, i))

    block_allocation_data.sort(reverse=True, key=lambda x: x[0])

    # Adjust memory budget based on model placement
    if vae_on_cpu and text_encoder_optimized:
        memory_overhead_ratio = 0.2  # 20% for gradients + optimizer
        print("VAE and text encoder optimized - maximizing GPU memory for model blocks")
    elif vae_on_cpu or text_encoder_optimized:
        memory_overhead_ratio = 0.3  # 30% overhead
        print("Some models optimized")
    else:
        memory_overhead_ratio = 0.4  # 40% for all models + training
        print("All models on GPU - conservative allocation")

    memory_budget = free_memory * (1.0 - memory_overhead_ratio) * 0.9  # 10% safety margin

    # Greedy allocation
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

    # Be more aggressive if models are offloaded
    if vae_on_cpu and text_encoder_optimized:
        blocks_to_swap = max(1, blocks_to_swap - 5)

    print(f"Auto-calculated blocks_to_swap: {blocks_to_swap}")
    print(f"Keeping {len(blocks) - blocks_to_swap}/{len(blocks)} blocks on GPU")
    print(f"Estimated model memory: {allocated_memory / (1024 ** 3):.1f}GB")

    return blocks_to_swap


def monkey_patch_offloading(config):
    """Monkey patch custom_offloading_utils to support auto_blocks_to_swap"""
    try:
        from modules.custom_offloading_utils import ModelOffloader
        original_init = ModelOffloader.__init__

        def patched_init(self, block_type, blocks, num_blocks, blocks_to_swap, supports_backward, device, debug=False):
            if config.get('train', {}).get('auto_blocks_to_swap', False) and blocks_to_swap is not None:
                optimal_blocks = calculate_optimal_blocks_from_model(blocks, device, config)
                print(f"Auto-calculated blocks_to_swap: {optimal_blocks} (was: {blocks_to_swap})")
                blocks_to_swap = optimal_blocks

            original_init(self, block_type, blocks, num_blocks, blocks_to_swap, supports_backward, device, debug)

        ModelOffloader.__init__ = patched_init
        print("Successfully monkey-patched ModelOffloader for auto_blocks_to_swap")
    except ImportError as e:
        print(f"Could not monkey patch offloading: {e}")



def run_command(cmd, description=""):
    """Execute command and handle errors"""
    print(f"\n{'=' * 60}")
    print(f"EXECUTING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)
    print(f"SUCCESS: {description} completed")


def build_args_from_dict(config_dict, prefix="--"):
    """Convert dictionary to command line arguments"""
    args = []
    for key, value in config_dict.items():
        if value is True:
            args.append(f"{prefix}{key}")
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            args.extend([f"{prefix}{key}"] + [str(v) for v in value])
        else:
            args.extend([f"{prefix}{key}", str(value)])
    return args


def get_script_path(framework, script_type, paths_config):
    """Get script path - use direct path if specified, otherwise build from framework"""
    script_key = f"{framework}_{script_type}_script"

    # Check if direct path is specified
    if script_key in paths_config:
        return paths_config[script_key]

    # Build default path from framework
    script_name = f"{framework}_{script_type}.py"
    return f"src/musubi_tuner/{script_name}"


def main():



    if len(sys.argv) != 2:
        print("Usage: python train.py path/to/config.toml")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"ERROR: Config file {config_path} not found")
        sys.exit(1)

    # Load config
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    # Apply monkey patch for auto_blocks_to_swap
    monkey_patch_offloading(config)

    # Get runner configuration
    runner = config.get('runner', {})
    framework = runner.get('framework', 'hv')
    skip_cache = runner.get('skip_cache', False)
    use_uv = runner.get('use_uv', True)
    cuda_extra = runner.get('cuda_extra', 'cu124')

    # Build base command prefix
    base_cmd = []
    if use_uv:
        base_cmd = ['uv', 'run', f'--extra', cuda_extra]
    else:
        base_cmd = ['python']

    # Get paths
    paths = config.get('paths', {})

    # Step 1: Cache latents (if not skipped)
    if not skip_cache:
        print(f"\n🔄 STEP 1: Caching latents for {framework.upper()}...")

        cache_script = get_script_path(framework, 'cache_latents', paths)
        cache_cmd = base_cmd + [cache_script]

        # Add cache-specific args
        cache_config = config.get('cache_latents', {})
        cache_args = build_args_from_dict(cache_config)

        run_command(cache_cmd + cache_args, "Latent caching")

    # Step 2: Cache text encoder outputs (if not skipped)
    if not skip_cache:
        print(f"\n🔄 STEP 2: Caching text encoder outputs for {framework.upper()}...")

        te_script = get_script_path(framework, 'cache_text_encoder_outputs', paths)
        te_cmd = base_cmd + [te_script]

        # Add text encoder cache args
        te_config = config.get('cache_text_encoder', {})
        te_args = build_args_from_dict(te_config)

        run_command(te_cmd + te_args, "Text encoder caching")

    # Step 3: Train
    print(f"\n🚀 STEP 3: Starting training with {framework.upper()}...")

    # Build accelerate command
    accelerate_config = config.get('accelerate', {})
    accelerate_args = build_args_from_dict(accelerate_config)

    # Get training script
    train_script = get_script_path(framework, 'train_network', paths)
    train_config = config.get('train', {})

    # Build training arguments
    train_args = build_args_from_dict(train_config)

    # Build final training command
    if use_uv:
        train_cmd = ['uv', 'run', f'--extra', cuda_extra, 'accelerate', 'launch'] + accelerate_args + [
            train_script] + train_args
    else:
        train_cmd = ['accelerate', 'launch'] + accelerate_args + [train_script] + train_args

    run_command(train_cmd, f"Training ({framework.upper()})")

    print("\n✅ TRAINING COMPLETE!")
    print(f"Check outputs in: {train_config.get('output_dir', 'outputs')}")


if __name__ == "__main__":
    main()
