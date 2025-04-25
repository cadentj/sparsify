# python -m sparsify unsloth/Qwen2.5-7B-Instruct /root/combined \
#     --layers 20 \
#     --grad_acc_steps 2 \
#     --ctx_len 2048 \
#     --batch_size 4 \
#     --expansion_factor 1 \
#     --k 64 \
#     --save_every 1500 \
#     --optimizer adam \
#     --clip_grad_norm 1.0 \
#     --lr 1e-3 \
#     --save_dir /root/ssaes \
#     --run_name qwen-ssae \
#     --subject_specific kh4dien/sae-Qwen2.5-7B-Instruct-6x


python -m sparsify unsloth/Qwen2.5-7B-Instruct /root/combined \
    --layers 20 \
    --grad_acc_steps 2 \
    --ctx_len 2048 \
    --batch_size 4 \
    --expansion_factor 6 \
    --k 128 \
    --save_every 1500 \
    --optimizer adam \
    --clip_grad_norm 1.0 \
    --save_dir /root/saes \
    --run_name qwen-ft \
    --finetune kh4dien/sae-Qwen2.5-7B-Instruct-6x \
    --lr 1e-3