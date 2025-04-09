#!/bin/bash

# Common parameters
export HF_TOKEN="your_huggingface_token_here"

# Array of models to test
MODELS=(
    "meta-llama/Llama-2-7b-hf"
    "mistralai/Mistral-7B-v0.1"
    "google/gemma-7b"
)

# Array of batch sizes to test
BATCH_SIZES=(8 16 32)

# Submit job for each combination
for model in "${MODELS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        echo "Submitting job for model: $model with batch size: $bs"
        sbatch \
            --export=MODEL_NAME="$model",BATCH_SIZE="$bs" \
            --job-name="${model##*/}_bs$bs" \
            job.slurm
    done
done