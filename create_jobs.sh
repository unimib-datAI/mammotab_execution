#!/bin/bash

# Load environment variables
source .env

# Split dataset into chunks
DATASET_FILE="work/mammotab_sample.jsonl"
TOTAL_ITEMS=$(wc -l < "$DATASET_FILE")
CHUNK_PREFIX="mammotab_chunk_"
split -l "$CHUNK_SIZE" -a 4 "$DATASET_FILE" "$CHUNK_PREFIX"

mkdir -p chunks
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
#SBATCH --account=datai
#SBATCH --partition=datai01
#SBATCH --job-name=${chunk%.jsonl}
#SBATCH --export=MODEL_NAME="$MODEL_NAME",BATCH_SIZE="$BATCH_SIZE",CHUNK_FILE="$chunk"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch_share/datai/dchieregato/job_logs/out_%x_%j.log
#SBATCH --error=/scratch_share/datai/dchieregato/job_logs/error_%x_%j.log
### Definitions
export BASEDIR="/scratch_share/datai/dchieregato"
export SHRDIR="/scratch_share/datai/dchieregato"
export LOCDIR="/scratch_local"
export TMPDIR=\$SHRDIR/tmp_\${SLURM_JOB_NAME}_\${SLURM_JOB_ID}

cd /scratch_share/datai/dchieregato/mammotab_execution

### Header
pwd; hostname; date

module purge
module load amd/slurm

set -a && source .env && set +a

echo "MODEL_NAME: $MODEL_NAME"
echo "HF_TOKEN: $HF_TOKEN"

source /scratch_share/datai/dchieregato/mammotab_execution/.venv/bin/activate

python work/test-model.py \
    --model_name "\$MODEL_NAME" \
    --hf_token "\$HF_TOKEN"

EOF

done
