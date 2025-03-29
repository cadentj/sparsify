# torchrun \
#     --nproc_per_node gpu -m sparsify google/gemma-3-4b-pt \
#     --layers 15 31 \
#     --distribute_modules \
#     --grad_acc_steps 8 \
#     --ctx_len 2048 \
#     --batch_size 2 \
#     --load_in_8bit \
#     --micro_acc_steps 2 \
#     --expansion_factor 8 \
#     --k 128 \
#     --save_every 1500 \
#     --split train[:20%]

uv run --active -m sparsify unsloth/Qwen2.5-Coder-32B-Instruct togethercomputer/RedPajama-Data-1T-Sample \
    --layers 31 47 \
    --grad_acc_steps 8 \
    --optimizer adam \
    --ctx_len 2048 \
    --batch_size 8 \
    --expansion_factor 8 \
    --k 128 \
    --save_every 1000 \
    --split train \
    --save_dir /root/saes \
    --max_examples 490000 \
    --val_dataset kh4dien/fineweb-sample \
    --val_max_examples 100 \
    --val_every 75 \
    --run_name gemma-3-4b