# Base Model + Emergent Data
python -m sparsify unsloth/Qwen2.5-7B-Instruct /root/medical \
    --layers 13 \
    --grad_acc_steps 8 \
    --ctx_len 2048 \
    --batch_size 8 \
    --expansion_factor 6 \
    --k 128 \
    --save_every 1500 \
    --optimizer adam \
    --clip_grad_norm 1.0 \
    --save_dir /root/saes \
    --run_name start-to-data \
    --finetune kh4dien/sae-Qwen2.5-7B-Instruct-6x

# Emergent Model + Base Data
python -m sparsify unsloth/Qwen2.5-7B-Instruct togethercomputer/RedPajama-Data-1T-Sample \
    --layers 13 \
    --grad_acc_steps 8 \
    --ctx_len 2048 \
    --batch_size 8 \
    --expansion_factor 6 \
    --k 128 \
    --save_every 1500 \
    --optimizer adam \
    --save_dir /root/saes \
    --run_name start-to-model \
    --finetune kh4dien/sae-Qwen2.5-7B-Instruct-6x \
    --peft_path /workspace/qwen-medical-7b/checkpoint-440 \
    --max_examples 10000

# (Base Model + Emergent Data) + Emergent Model
python -m sparsify unsloth/Qwen2.5-7B-Instruct /root/medical \
    --layers 13 \
    --grad_acc_steps 8 \
    --ctx_len 2048 \
    --batch_size 8 \
    --expansion_factor 6 \
    --k 128 \
    --save_every 1500 \
    --optimizer adam \
    --save_dir /root/saes \
    --run_name start-to-data-to-final \
    --finetune /root/saes/start-to-data/layers.13 \
    --peft_path /workspace/qwen-medical-7b/checkpoint-440

# (Emergent Model + Base Data) + Emergent Data
python -m sparsify unsloth/Qwen2.5-7B-Instruct /root/medical \
    --layers 13 \
    --grad_acc_steps 8 \
    --ctx_len 2048 \
    --batch_size 8 \
    --expansion_factor 6 \
    --k 128 \
    --save_every 1500 \
    --optimizer adam \
    --clip_grad_norm 1.0 \
    --save_dir /root/saes \
    --run_name start-to-model-to-final \
    --finetune /root/saes/start-to-model/model.model.layers.13 \
    --peft_path /workspace/qwen-medical-7b/checkpoint-440