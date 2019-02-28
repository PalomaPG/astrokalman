#!/bin/bash
#SBATCH --job-name="phuente_candidate_search"
#SBATCH --partition=slims
#SBATCH -c 4
#SBATCH --mem-per-cpu=2400
#SBATCH --output=outputs/CS-%j.out
#SBATCH --error=outputs/CS-%j.err
#SBATCH --array=0-2975
#SBATCH --exclusive=user
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paloma.perez.gar@gmail.com
module load astro
echo $SLURM_ARRAY_TASK_ID;
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory="$SLURM_SUBMIT_DIR
python candidate_search.py $SLURM_ARRAY_TASK_ID
echo "Trabajo terminado";
