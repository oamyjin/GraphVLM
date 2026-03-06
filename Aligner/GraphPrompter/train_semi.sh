#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00  # 可选，设置最大运行时间

# 载入模块和激活环境
module purge
module load miniconda
source activate prompter

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

# 遍历数据集列表
for dataset in 'cora_sup' 'cora_semi'
do
    echo "Training on dataset: $dataset"
    python train.py --dataset $dataset --model_name graph_llm --patience 2 --num_epochs 15
done
