#!/bin/bash

# Load environment variables
source .env

# Split dataset into chunks
DATASET_FILE="work/mammotab_sample.jsonl"
TOTAL_ITEMS=$(wc -l < "$DATASET_FILE")
CHUNK_PREFIX="mammotab_chunk_"
split -l "$CHUNK_SIZE" -a 4 "$DATASET_FILE" "$CHUNK_PREFIX"

# Rename chunks to include .jsonl suffix
for chunk in ${CHUNK_PREFIX}*; do
    mv "$chunk" "chunks/$chunk.jsonl"
done

# Create job files and submit
for chunk in chunks/${CHUNK_PREFIX}*.jsonl; do
    JOB_FILE="chunks/job_${chunk##*/}.slurm"
    
    # Generate job file
    cat << EOF > "$JOB_FILE"
#!/bin/bash
#SBATCH --job-name=${chunk%.jsonl}
#SBATCH --export=MODEL_NAME="$MODEL_NAME",BATCH_SIZE="$BATCH_SIZE",CHUNK_FILE="$chunk"

srun python work/generate.py \
    --model_name "\$MODEL_NAME" \
    --batch_size "\$BATCH_SIZE" \
    --hf_token "\$HF_TOKEN" \
    --input_file "\$CHUNK_FILE"
EOF

    # Submit job
    #sbatch "$JOB_FILE"
done