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
#SBATCH --account=m.cremaschi
#SBATCH --partition=only-one-gpu
#SBATCH --job-name=${chunk%.jsonl}
#SBATCH --export=MODEL_NAME="$MODEL_NAME",BATCH_SIZE="$BATCH_SIZE",CHUNK_FILE="$chunk"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=job_logs/out_%x_%j.log # Standard output and error log, with job name and id
#SBATCH --error=job_logs/error_%x_%j.log
### Definitions
export BASEDIR="/home/m.cremaschi/mammotab_execution"
export SHRDIR="/scratch_share/datai/`whoami`"
export LOCDIR="/scratch_local"
export TMPDIR=$SHRDIR/$BASEDIR/tmp_$SLURM_JOB_NAME_$SLURM_JOB_ID

cd /home/m.cremaschi/mammotab_execution/

### Header
pwd; hostname; date

module purge
module load amd/slurm

source /home/m.cremaschi/.bashrc
conda activate python3.11

torchrun --nproc-per-node=1 --standalone python work/generate.py \
    --model_name "\$MODEL_NAME" \
    --batch_size "\$BATCH_SIZE" \
    --hf_token "\$HF_TOKEN" \
    --input_file "\$CHUNK_FILE"

EOF

    # Submit job
    # sbatch "$JOB_FILE"
done