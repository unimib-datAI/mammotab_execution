#!/bin/bash

# --- Configuration ---
MAX_CONCURRENT_JOBS=5       # Maximum number of jobs allowed in queue/running at once
JOB_DIR="chunks"            # Directory containing the .slurm job files
CHECK_INTERVAL=60           # Seconds to wait between checking the queue
SUBMITTED_LOG="submitted_jobs.log" # File to log submitted job filenames

# --- Ensure job directory exists ---
if [[ ! -d "$JOB_DIR" ]]; then
  echo "Error: Job directory '$JOB_DIR' not found."
  exit 1
fi

# --- Get list of job files ---
# Find all .slurm files, excluding any already listed in the log file
find "$JOB_DIR" -maxdepth 1 -name '*.slurm' -print0 | while IFS= read -r -d $'\0' job_file; do
  if ! grep -Fxq "$job_file" "$SUBMITTED_LOG" &>/dev/null; then
    PENDING_JOBS+=("$job_file")
  fi
done

if [[ ${#PENDING_JOBS[@]} -eq 0 ]]; then
  echo "No pending .slurm files found in '$JOB_DIR' that aren't already logged in '$SUBMITTED_LOG'."
  exit 0
fi

echo "Found ${#PENDING_JOBS[@]} job(s) to submit."

# --- Main submission loop ---
while [[ ${#PENDING_JOBS[@]} -gt 0 ]]; do
  # Get current number of user's jobs (Pending or Running)
  # -h: no header, -t PD,R: states Pending,Running, -u $USER: your jobs
  current_jobs=$(squeue -u "$USER" -h -t PD,R | wc -l)
  echo "Currently $current_jobs jobs active (limit: $MAX_CONCURRENT_JOBS)."

  # Calculate how many jobs can be submitted
  available_slots=$((MAX_CONCURRENT_JOBS - current_jobs))

  if [[ $available_slots -gt 0 ]]; then
    # Determine how many jobs to submit in this batch
    num_to_submit=${#PENDING_JOBS[@]}
    if [[ $num_to_submit -gt $available_slots ]]; then
      num_to_submit=$available_slots
    fi
    echo "Attempting to submit $num_to_submit job(s)..."

    submitted_in_batch=0
    temp_pending=() # Array to hold jobs remaining after this batch

    for (( i=0; i<${#PENDING_JOBS[@]}; i++ )); do
      job_file="${PENDING_JOBS[$i]}"
      if [[ $submitted_in_batch -lt $num_to_submit ]]; then
        echo "Submitting: $job_file"
        sbatch_output=$(sbatch "$job_file" 2>&1) # Capture stdout and stderr
        sbatch_exit_code=$?

        if [[ $sbatch_exit_code -eq 0 ]]; then
          echo "Successfully submitted: $sbatch_output"
          # Log the submitted job file to avoid resubmission on script restart
          echo "$job_file" >> "$SUBMITTED_LOG"
          ((submitted_in_batch++))
        else
          echo "Error submitting $job_file: $sbatch_output"
          # Keep the job in the pending list to retry later
          temp_pending+=("$job_file")
          # Optional: Exit if a persistent error occurs?
          # if [[ "$sbatch_output" != *"QOSMaxSubmitJobPerUserLimit"* ]]; then
          #   echo "Exiting due to non-limit error."
          #   exit 1
          # fi
        fi
      else
        # Add remaining jobs to the temporary list
        temp_pending+=("$job_file")
      fi
    done
    # Update the PENDING_JOBS array
    PENDING_JOBS=("${temp_pending[@]}")

  else
    echo "No available slots. Waiting ${CHECK_INTERVAL} seconds..."
    sleep "$CHECK_INTERVAL"
  fi

  # If there are still jobs left, wait before the next check
  if [[ ${#PENDING_JOBS[@]} -gt 0 ]]; then
     # Add a small delay even if we just submitted, allowing Slurm state to update
     if [[ $submitted_in_batch -gt 0 ]]; then
       sleep 5
     else
       # If we didn't submit anything (because queue was full), wait the full interval
       sleep "$CHECK_INTERVAL"
     fi
  fi
done

echo "All jobs submitted."
exit 0
