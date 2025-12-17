#!/usr/bin/env python3
"""
Register DeepSeek-V2-Lite TrainSpec for TorchTitan.

This script registers a custom TrainSpec for DeepSeek-V2-Lite 16B model.
Since DeepSeek-V2-Lite is similar to DeepSeek-V3 (both are MoE models),
we reuse the DeepSeek-V3 TrainSpec components.

Usage:
    # Import this module before running torchtitan.train
    import register_deepseek_v2_lite
    
    # Or run directly to register
    python register_deepseek_v2_lite.py
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.deepseek_v3 import (
    DeepSeekV3Model,
    DeepSeekV3ModelArgs,
    parallelize_deepseekv3,
)
from torchtitan.models.deepseek_v3.model.state_dict_adapter import (
    DeepSeekV3StateDictAdapter,
)
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

# DeepSeek-V2-Lite 16B model arguments
# Note: These are based on DeepSeek-V3 16B architecture since V2-Lite is similar
# You may need to adjust these based on the actual DeepSeek-V2-Lite specifications
deepseek_v2_lite_args = {
    "16B": DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=27,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        attn_type="flex",
        attn_mask_type="block_causal",
    ),
}


def register_deepseek_v2_lite():
    """Register the DeepSeek-V2-Lite TrainSpec."""
    train_spec = TrainSpec(
        model_cls=DeepSeekV3Model,
        model_args=deepseek_v2_lite_args,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
    
    register_train_spec("deepseek_v2_lite", train_spec)
    print("✓ Successfully registered TrainSpec: deepseek_v2_lite")


if __name__ == "__main__":
    register_deepseek_v2_lite()
    print("\nTrainSpec 'deepseek_v2_lite' is now registered and ready to use.")
    print("You can now use 'name = \"deepseek_v2_lite\"' in your train_config.toml")
else:
    # Auto-register when imported
    register_deepseek_v2_lite()

