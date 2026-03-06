#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="apllaga"
#SBATCH --output=log_loaddata/%j_reddit.out
#SBATCH --error=log_loaddata/%j_reddit.err
#SBATCH --account=pr_142_tandon_priority

module purge

singularity exec --nv --overlay /scratch/jl11523/mgllm2/overlay-50G-10M.ext3:ro \
    /scratch/jl11523/mgllm2/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate llaga3

python  ../dataloader/data_preprocess.py

"