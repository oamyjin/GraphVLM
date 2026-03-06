#!/usr/bin/env bash
#
# A stand-alone script to run the Python benchmarking tasks.

# If SLURM_JOB_ID is not set in your environment, define a fallback:
SLURM_JOB_ID=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}

# -----------------------------------------------------------------------------
# Declare model types
MODEL_LIST=('MLP' 'GCN' 'SAGE')

# Declare feature types
FEAT_LIST=('clip' 'clipimage' 'clipnonstruc' 'clipnonstruc')

# Declare datasets
DATASET_LIST=('Movies' 'Toys' 'Grocery' 'Arts' 'CD')

# Name for the master results file
RESULT_FILE="${SLURM_JOB_ID}_results.log"

# Create/clear the result file
touch "$RESULT_FILE"

# -----------------------------------------------------------------------------
# Loop over each dataset
for DATASET in "${DATASET_LIST[@]}"
do
    # Loop over each model type
    for MODEL in "${MODEL_LIST[@]}"
    do
        # Loop over each feature type
        for FEAT in "${FEAT_LIST[@]}"
        do
            python main.py --config-name defaults \
                model_name="$MODEL" \
                dataset="$DATASET" \
                result_file="$RESULT_FILE" \
                feat="$FEAT" \
                > "${SLURM_JOB_ID}_${DATASET}-${MODEL}-${FEAT}.log" 2>&1
        done
    done
done
