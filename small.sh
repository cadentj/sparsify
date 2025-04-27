# python -m sparsify unsloth/Qwen2.5-7B-Instruct togethercomputer/RedPajama-Data-1T-Sample \
#     --layers 20 \
#     --grad_acc_steps 16 \
#     --ctx_len 2048 \
#     --batch_size 4 \
#     --expansion_factor 6 \
#     --k 128 \
#     --save_every 1500 \
#     --optimizer adam \
#     --clip_grad_norm 1.0 \
#     --save_dir /workspace/saes \
#     --run_name qwen-7b \

python -m sparsify unsloth/Qwen2.5-7B-Instruct togethercomputer/RedPajama-Data-1T-Sample \
    --layers 6 13 \
    --grad_acc_steps 8 \
    --ctx_len 2048 \
    --batch_size 8 \
    --expansion_factor 6 \
    --k 128 \
    --save_every 1500 \
    --optimizer adam \
    --clip_grad_norm 1.0 \
    --save_dir /workspace/saes \
    --run_name qwen-7b \
    --max_examples 192_000

# python -m sparsify unsloth/Qwen2.5-Coder-32B-Instruct /root/combined \
#     --layers 31 \
#     --grad_acc_steps 4 \
#     --ctx_len 2048 \
#     --batch_size 1 \
#     --expansion_factor 1 \
#     --k 128 \
#     --save_every 1500 \
#     --optimizer adam \
#     --clip_grad_norm 1.0 \
#     --save_dir /root/ssaes \
#     --run_name qwen-combined \
#     --subject_specific /workspace/qwen-saes-two/qwen-step-final/model.layers.31


# python -m sparsify unsloth/Qwen2.5-Coder-32B-Instruct /root/combined \
#     --layers 31 \
#     --grad_acc_steps 8 \
#     --ctx_len 2048 \
#     --batch_size 1 \
#     --expansion_factor 8 \
#     --k 128 \
#     --save_every 1500 \
#     --optimizer adam \
#     --clip_grad_norm 1.0 \
#     --save_dir /root/ssaes \
#     --run_name qwen-ft-high-lr \
#     --finetune /workspace/qwen-saes-two/qwen-step-final/model.layers.31 \
#     --lr 1e-3