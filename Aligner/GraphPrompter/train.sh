#!/bin/bash

module purge
module load miniconda
source activate gpt

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
# 'cora_sup' 'pubmed_sup' 'arxiv_sup' 'products_sup'

nvidia-smi

# for dataset in 'cora_sup' 'cora_semi'
# for dataset in 'cora_semi'
# do
#     echo "Training on dataset: $dataset"
#     python train.py --dataset $dataset --model_name graph_llm --patience 2 --num_epochs 15
# done

# 'movies' 'grocery' 'toys' 'reddit'
for dataset in 'cd'
do
    echo "Training on dataset: $dataset"
    python train.py --dataset $dataset --model_name graph_llm --patience 2 --num_epochs 15
done