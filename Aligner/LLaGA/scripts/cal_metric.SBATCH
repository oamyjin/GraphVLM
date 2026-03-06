#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="ap-l-ev"
#SBATCH --output=log_metric/%j_all_aug23.out
#SBATCH --error=log_metric/%j_all_aug23.err
#SBATCH --account=pr_142_tandon_advanced

module purge

singularity exec --nv --overlay /scratch/jl11523/mgllm/overlay-50G-10M.ext3:ro \
    /scratch/jl11523/mgllm/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate llaga

python ../eval/eval_res.py \
--dataset mgllm \
--task nc  \
--res_path ../eval_output \

"