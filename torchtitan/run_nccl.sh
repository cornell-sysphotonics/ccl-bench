unset LD_PRELOAD
unset MSCCLPP_XML_FILE
unset MSCCLPP_DEBUG

CONFIG=/pscratch/sd/x/xz987/CS5470/final_project/torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml

torchrun --nproc_per_node=4 --standalone \
    -m torchtitan.train --job.config_file $CONFIG