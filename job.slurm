#!/bin/bash
#SBATCH --job-name=llm_job
#SBATCH --output=output_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Load required modules
module purge
module load CUDA/11.8
module load cuDNN/8.6

# Set environment variables
export HF_TOKEN=${HF_TOKEN}
export MODEL_NAME=${MODEL_NAME}
export TOKENIZER_NAME=${TOKENIZER_NAME:-$MODEL_NAME}
export BATCH_SIZE=${BATCH_SIZE:-8}
export CHUNK_SIZE=${CHUNK_SIZE:-64}

# Run the script
python -u work/main.py