#!/bin/bash

python ../llava/eval/run_llava.py \
    --model-path ../local_model/llava-v1.5-7b \
    --cvs-file ../dataset/Movies.csv \
    --start-idx 2766 \
    --true-label-file ../dataset/true_labels_Movies.csv \
    --write-true-label-file False \
    --use-text True \
    --output-file ../eval-results/movies/imagetext/output_movies_imagetext_from2766.json \
    --temperature 0

python cal_metric.py

