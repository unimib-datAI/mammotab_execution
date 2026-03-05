#!/bin/bash
#SBATCH --account=datai
#SBATCH --partition=datai01
#SBATCH --job-name=test-model
#SBATCH --export=MODEL_NAME="$MODEL_NAME",BATCH_SIZE="$BATCH_SIZE",CHUNK_FILE="$chunk"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch_share/datai/dchieregato/job_logs/out_%x_%j.log
#SBATCH --error=/scratch_share/datai/dchieregato/job_logs/error_%x_%j.log
### Definitions
export BASEDIR="/scratch_share/datai/dchieregato"
export SHRDIR="/scratch_share/datai/m.cremaschi"
export LOCDIR="/scratch_local"
export TMPDIR=\$SHRDIR/\$BASEDIR/tmp_\${SLURM_JOB_NAME}_\${SLURM_JOB_ID}

cd /scratch_share/datai/dchieregato/

### Header
pwd; hostname; date

module purge
module load amd/slurm

conda activate python3.11

set -a && source .env && set +a

echo "MODEL_NAME: $MODEL_NAME"
echo "HF_TOKEN: $HF_TOKEN"

source /scratch_share/datai/dchieregato/mammotab_execution/.venv/bin/activate

python work/test-model.py \
    --model_name "$MODEL_NAME" \
    --hf_token "$HF_TOKEN"
