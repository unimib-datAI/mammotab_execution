#!/bin/bash
#SBATCH --account=datai
#SBATCH --partition=datai01
#SBATCH --job-name=test-model
#SBATCH --export=MODEL_NAME="$MODEL_NAME",TOKENIZER_NAME="$TOKENIZER_NAME",ADAPTER_PATH="$ADAPTER_PATH",LOAD_IN_4BIT="$LOAD_IN_4BIT",LOAD_IN_8BIT="$LOAD_IN_8BIT",OFFLOAD_DIR="$OFFLOAD_DIR",MODEL_DTYPE="$MODEL_DTYPE",BATCH_SIZE="$BATCH_SIZE",CHUNK_FILE="$chunk"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch_share/datai/dchieregato/mammotab_execution/job_logs/out_%x_%j.log
#SBATCH --error=/scratch_share/datai/dchieregato/mammotab_execution/job_logs/error_%x_%j.log
### Definitions
export HF_HOME="/scratch_share/datai/dchieregato"
export BASEDIR="/scratch_share/datai/dchieregato"
export SHRDIR="/scratch_share/datai/dchieregato"
export LOCDIR="/scratch_local"
export TMPDIR=$SHRDIR/tmp_${SLURM_JOB_NAME}_${SLURM_JOB_ID}

cd /scratch_share/datai/dchieregato/mammotab_execution

### Header
pwd; hostname; date

module purge
module load amd/slurm

set -a && source .env && set +a
export OFFLOAD_DIR="${OFFLOAD_DIR:-$TMPDIR/model_offload}"
mkdir -p "$OFFLOAD_DIR"

echo "MODEL_NAME: $MODEL_NAME"
echo "TOKENIZER_NAME: $TOKENIZER_NAME"
echo "ADAPTER_PATH: $ADAPTER_PATH"
echo "LOAD_IN_4BIT: $LOAD_IN_4BIT"
echo "LOAD_IN_8BIT: $LOAD_IN_8BIT"
echo "OFFLOAD_DIR: $OFFLOAD_DIR"
echo "MODEL_DTYPE: $MODEL_DTYPE"
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN: set"
else
    echo "HF_TOKEN: not set"
fi

source /scratch_share/datai/dchieregato/mammotab_execution/.venv/bin/activate

python work/test-model.py \
    --model_name "$MODEL_NAME" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --adapter_path "$ADAPTER_PATH" \
    --load_in_4bit "$LOAD_IN_4BIT" \
    --load_in_8bit "$LOAD_IN_8BIT" \
    --offload_dir "$OFFLOAD_DIR" \
    --model_dtype "$MODEL_DTYPE" \
    --hf_token "$HF_TOKEN"
