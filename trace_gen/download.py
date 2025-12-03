#!/usr/bin/env python3
"""
Script to pre-download and cache Hugging Face models and tokenizers
for offline use with vLLM.

Usage:
    python preload_model.py --model-id microsoft/DialoGPT-medium
    python preload_model.py --model-id meta-llama/Llama-2-7b-chat-hf --token YOUR_HF_TOKEN
    python preload_model.py --model-id microsoft/DialoGPT-medium --cache-dir /path/to/custom/cache
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

def preload_model_and_tokenizer(
    model_id: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    force_download: bool = False
):
    """
    Pre-download model and tokenizer to local cache.
    
    Args:
        model_id: Hugging Face model identifier (e.g., 'microsoft/DialoGPT-medium')
        cache_dir: Custom cache directory path (optional)
        token: Hugging Face authentication token (optional)
        trust_remote_code: Whether to trust remote code in model repo
        force_download: Whether to force re-download even if cached
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        import torch
    except ImportError as e:
        print(f"Error: Required packages not installed. Please run:")
        print("pip install transformers torch")
        sys.exit(1)

    # Set cache directory if provided
    if cache_dir:
        cache_dir = Path(cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / 'transformers')
        os.environ['HF_DATASETS_CACHE'] = str(cache_dir / 'datasets')
        print(f"Using custom cache directory: {cache_dir}")

    print(f"Pre-loading model and tokenizer: {model_id}")
    print(f"Cache location: {os.environ.get('TRANSFORMERS_CACHE', '~/.cache/huggingface/transformers')}")

    # Common arguments for all downloads
    common_kwargs = {
        'trust_remote_code': trust_remote_code,
        'token': token,
        'force_download': force_download,
    }
    
    if cache_dir:
        common_kwargs['cache_dir'] = str(cache_dir / 'transformers')

    try:
        # Step 1: Download config
        print("\n1. Downloading model configuration...")
        config = AutoConfig.from_pretrained(model_id, **common_kwargs)
        print(f"✓ Config downloaded successfully")
        print(f"  Model type: {config.model_type}")
        print(f"  Architecture: {config.architectures}")

        # Step 2: Download tokenizer
        print("\n2. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
        print(f"✓ Tokenizer downloaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Model max length: {getattr(tokenizer, 'model_max_length', 'Unknown')}")

        # Step 3: Download generation config (needed by vLLM)
        print(f"\n3. Downloading generation config...")
        try:
            from transformers import GenerationConfig
            gen_config = GenerationConfig.from_pretrained(model_id, **common_kwargs)
            print(f"✓ Generation config downloaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not download generation config: {e}")
            print("  This is normal for some models that don't have generation_config.json")

        # Step 4: Download all model files (weights, etc.)
        print(f"\n4. Downloading model weights and additional files...")
        print("   This may take several minutes for large models...")
        
        # Download all files by using huggingface_hub directly
        try:
            from huggingface_hub import snapshot_download
            print("   Using snapshot_download to get all model files...")
            
            snapshot_kwargs = {
                'repo_id': model_id,
                'repo_type': 'model',
                'local_dir_use_symlinks': False,
                'token': token,
            }
            if cache_dir:
                snapshot_kwargs['cache_dir'] = str(cache_dir / 'hub')
            
            local_dir = snapshot_download(**snapshot_kwargs)
            print(f"✓ All model files downloaded to: {local_dir}")
            
        except Exception as e:
            print(f"⚠ Snapshot download failed, trying model loading: {e}")
            # Fallback to model loading
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16,  # Use float16 to save memory
                    low_cpu_mem_usage=True,     # Use less CPU memory
                    device_map="cpu",           # Keep on CPU
                    **common_kwargs
                )
                print(f"✓ Model weights downloaded successfully")
                
                # Get some basic model info
                num_parameters = sum(p.numel() for p in model.parameters())
                print(f"  Total parameters: {num_parameters:,}")
                print(f"  Model size: ~{num_parameters * 2 / 1e9:.1f}GB (float16)")
                
                # Clean up memory
                del model
                
            except Exception as e2:
                print(f"⚠ Warning: Could not load model for inspection: {e2}")
                print("  Model files may still have been downloaded to cache")

        # Step 5: Verify files are in cache
        print(f"\n4. Verifying cache contents...")
        if cache_dir:
            cache_path = cache_dir / 'transformers'
        else:
            cache_path = Path.home() / '.cache' / 'huggingface' / 'transformers'
        
        model_cache_dirs = list(cache_path.glob(f"*{model_id.replace('/', '--')}*"))
        if model_cache_dirs:
            print(f"✓ Found {len(model_cache_dirs)} cached directories")
            for cache_dir_path in model_cache_dirs[:3]:  # Show first 3
                files = list(cache_dir_path.glob("*"))
                print(f"  {cache_dir_path.name}: {len(files)} files")
                if len(model_cache_dirs) > 3:
                    print(f"  ... and {len(model_cache_dirs) - 3} more directories")
                break
        else:
            print("⚠ No cache directories found (this might be normal)")

        print(f"\n🎉 Successfully pre-loaded {model_id}!")
        
        if cache_dir:
            print(f"    # vLLM will automatically find the cached model")
            print(f"    # You can also set HF_HOME={cache_dir} environment variable")

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print(f"\nTroubleshooting tips:")
        print(f"1. Check if the model ID is correct: https://huggingface.co/{model_id}")
        print(f"2. For private models, make sure you have access and provide --token")
        print(f"3. Try with --trust-remote-code if the model requires it")
        print(f"4. Check your internet connection")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download Hugging Face models and tokenizers for offline vLLM usage"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model identifier (e.g., 'microsoft/DialoGPT-medium')"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory path (default: ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token for private models"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in model repository"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if model is already cached"
    )

    args = parser.parse_args()

    # Get token from environment if not provided
    if not args.token:
        args.token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')

    print("=" * 60)
    print("🤗 Hugging Face Model Pre-loader for vLLM")
    print("=" * 60)
    
    preload_model_and_tokenizer(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        force_download=args.force_download
    )


if __name__ == "__main__":
    main()