#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="lg-arts"
#SBATCH --output=log/%j_arts_clip_aug23.out
#SBATCH --error=log/%j_arts_clip_aug23.err
#SBATCH --account=pr_142_tandon_advanced

module purge

singularity exec --nv --overlay /scratch/jl11523/mgllm2/overlay-50G-10M.ext3:ro \
    /scratch/jl11523/mgllm2/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate llaga3

wandb offline

DATASET=Arts
FEAT_LIST=('augnonstruc2' 'augnonstruc3')

for FEAT in \"\${FEAT_LIST[@]}\"
do

    python ../train/train_mem.py \
    --model_name_or_path /scratch/jl11523/projects/LLaGA/base_model/vicuna-7b-v1.5-16k \
    --version v1 \
    --cache_dir  ../../checkpoint \
    --pretrained_embedding_type \$FEAT \
    --tune_mm_mlp_adapter True \
    --mm_use_graph_start_end False \
    --mm_use_graph_patch_token False \
    --bf16 True \
    --output_dir ../model_finetuned/\$DATASET/llaga-vicuna-7b-\$FEAT-2-10-linear-projector_nc \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \
    --use_hop 2 \
    --sample_neighbor_size 10 \
    --mm_projector_type "linear" \
    --use_task nc \
    --use_dataset \$DATASET \
    --template "ND" \
    > log_train/${SLURM_JOB_ID}_\${DATASET}_\${FEAT}.log 2> log_train/${SLURM_JOB_ID}_\${DATASET}_\${FEAT}.err


    python ../eval/eval_pretrain.py \
    --model_path ../model_finetuned/\$DATASET/llaga-vicuna-7b-\$FEAT-2-10-linear-projector_nc \
    --model_base /scratch/jl11523/projects/LLaGA/base_model/vicuna-7b-v1.5-16k \
    --conv_mode v1 \
    --dataset \$DATASET \
    --pretrained_embedding_type \$FEAT \
    --use_hop 2 \
    --sample_neighbor_size 10 \
    --answers_file /scratch/jl11523/projects/LLaGA/eval_output/\$DATASET/\${DATASET}_\${FEAT}_label.jsonl \
    --task nc \
    --cache_dir ../../checkpoint \
    --template ND \
    > log_eval/${SLURM_JOB_ID}_\${DATASET}_\${FEAT}.log 2> log_eval/${SLURM_JOB_ID}_\${DATASET}_\${FEAT}.err

done

"