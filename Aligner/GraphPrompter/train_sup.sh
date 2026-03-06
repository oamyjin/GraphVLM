#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00  # 可选，设置最大运行时间

# 载入模块和激活环境
module purge
module load miniconda
source activate prompter

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
# 'cora_sup' 'pubmed_sup' 'arxiv_sup' 'products_sup'
# 遍历数据集列表
for dataset in 'arxiv_sup'
do
    echo "Training on dataset: $dataset"
    python train.py --dataset $dataset --model_name graph_llm --num_epochs 15 --patience 2 --output yuanfu_output
done
