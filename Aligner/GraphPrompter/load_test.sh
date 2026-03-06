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
source activate gpt

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

# # 遍历数据集列表
# for dataset in "amazon-sports" "amazon-computers" "amazon-photo"
# do
#     echo "Processing dataset: $dataset"
#     python label_mapping.py --dataset $dataset
# done

# 遍历数据集列表
for dataset in "amazon-sports"
do
    echo "Processing dataset: $dataset"
    python label_mapping.py --dataset $dataset
done

