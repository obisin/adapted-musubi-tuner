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

import torch
torch.cuda.empty_cache()

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

        if key == 'auto_blocks_to_swap':
            continue

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

    # Apply monkey patch for auto_blocks_to_swap
    # patch_offloading(config)

    if config.get('train', {}).get('auto_blocks_to_swap', False):
        os.environ['AUTO_BLOCKS_TO_SWAP'] = 'true'

        # Force VAE and text encoder optimizations when auto is enabled
        train_config = config.get('train', {})
        cache_latents_config = config.get('cache_latents', {})
        cache_te_config = config.get('cache_text_encoder', {})

        # Set vae_cache_cpu if not already set
        if 'vae_cache_cpu' not in train_config:
            train_config['vae_cache_cpu'] = True

        # Set fp8_t5 if not already set
        if 'fp8_t5' not in train_config:
            train_config['fp8_t5'] = True

    run_command(train_cmd, f"Training ({framework.upper()})")

    print("\n✅ TRAINING COMPLETE!")
    print(f"Check outputs in: {train_config.get('output_dir', 'outputs')}")


if __name__ == "__main__":
    main()
