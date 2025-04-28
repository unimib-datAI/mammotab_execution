#!/bin/bash
#SBATCH --account=m.cremaschi
#SBATCH --partition=only-one-gpu
#SBATCH --job-name=test-model
#SBATCH --export=MODEL_NAME="$MODEL_NAME",BATCH_SIZE="$BATCH_SIZE",CHUNK_FILE="$chunk"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/m.cremaschi/mammotab_execution/job_logs/out_%x_%j.log 
#SBATCH --error=/home/m.cremaschi/mammotab_execution/job_logs/error_%x_%j.log
### Definitions
export BASEDIR="/home/m.cremaschi/mammotab_execution"
export SHRDIR="/scratch_share/datai/m.cremaschi"
export LOCDIR="/scratch_local"
export TMPDIR=\$SHRDIR/\$BASEDIR/tmp_\${SLURM_JOB_NAME}_\${SLURM_JOB_ID}

cd /home/m.cremaschi/mammotab_execution/

### Header
pwd; hostname; date

module purge
module load amd/slurm

source /home/m.cremaschi/.bashrc
conda activate python3.11

set -a && source .env && set +a

echo "MODEL_NAME: $MODEL_NAME"
echo "HF_TOKEN: $HF_TOKEN"

torchrun --nproc-per-node=1 --standalone work/test-model.py \
    --model_name "\$MODEL_NAME" \
    --hf_token "\$HF_TOKEN"