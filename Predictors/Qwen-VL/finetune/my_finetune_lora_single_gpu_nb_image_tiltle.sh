#!/usr/bin/env bash
#
# A self-contained script to run Qwen-VL finetuning.

# Export environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0

# Define variables
DATASET="Movies"
NB_TYPE="nb_images_titles"
TOP_K=1
DATA_PATH="/scratch/jl11523/projects/Qwen-VL/finetune/fintune_dataset/structure_aware/${DATASET}_finetune_top${TOP_K}_${NB_TYPE}.json"
OUTPUT_QWEN="/scratch/jl11523/projects/Qwen-VL/local_model/finetuned_jan19/${DATASET}_finetuned_${NB_TYPE}_top${TOP_K}"

# Run the finetuning
python finetune.py \
    --model_name_or_path /scratch/jl11523/projects/Qwen-VL/local_model/Qwen-VL-Chat \
    --data_path "$DATA_PATH" \
    --bf16 True \
    --fix_vit True \
    --output_dir "$OUTPUT_QWEN" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --gradient_checkpointing \
    --use_lora
