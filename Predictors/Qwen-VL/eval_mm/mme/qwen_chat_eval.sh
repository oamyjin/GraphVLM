#!/usr/bin/env bash
#
# This script evaluates the model using different feature types for the "Movies" dataset.
# It runs eval-vl-chat.py and cal_metric.py for each feature type.

# If SLURM_JOB_ID is not set (e.g., running locally), use a timestamp fallback.
SLURM_JOB_ID=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}

DATASET="Movies"
FEAT_TYPE_LIST=('nb_images_titles' 'nb_images' 'nb_titles')

for FEAT_TYPE in "${FEAT_TYPE_LIST[@]}"
do
    IMAGE_PATH="/scratch/jl11523/dataset-mgllm/${DATASET}/images"
    MODEL="/scratch/jl11523/projects/Qwen-VL/local_model/Qwen-VL-Chat"
    ADAPTER="/scratch/jl11523/projects/Qwen-VL/local_model/finetuned_v3/${DATASET}_finetuned_${FEAT_TYPE}_top3"
    PROMPT="/scratch/jl11523/projects/Qwen-VL/eval_mm/mme/eval_prompt_files/${DATASET}_test_top3_${FEAT_TYPE}.json"
    OUTPUT="/scratch/jl11523/projects/Qwen-VL/eval-results-fintuned/${DATASET}/output_${DATASET}_${FEAT_TYPE}.json"

    # Create necessary output directory (if it doesn't exist)
    mkdir -p "/scratch/jl11523/projects/Qwen-VL/eval-results-fintuned/${DATASET}"

    # Run evaluation
    python eval-vl-chat.py \
        --model-path "$MODEL" \
        --path_to_adapter "$ADAPTER" \
        --cvs-file "/scratch/jl11523/projects/LLaVA/dataset/${DATASET}.csv" \
        --prompt-file "$PROMPT" \
        --image_path "$IMAGE_PATH" \
        --output-file "$OUTPUT"

    # Calculate metrics and save log
    python cal_metric.py \
        --dataset "${DATASET}" \
        --feat_type "${FEAT_TYPE}" \
        --result_file "$OUTPUT" \
        > "log_metric_v3/${SLURM_JOB_ID}_${DATASET}-${FEAT_TYPE}_finetuned.log" 2>&1
done
