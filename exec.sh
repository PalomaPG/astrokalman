#!/bin/bash
#SBATCH --job-name="candidate_search"
#SBATCH --partition=slims
#SBATCH -c 4
#SBATCH --mem-per-cpu=2400
#SBATCH --output=outputs/CS-%j.out
#SBATCH --error=outputs/CS-%j.err
#SBATCH --mail-user=paloma.perez.gar@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=0-2975
#SBATCH --exclusive=user

echo $SLURM_ARRAY_TASK_ID
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory="$SLURM_SUBMIT_DIR

module load Lmod/6.5
source $LMOD_PROFILE
ml icc/2017.4.196-GCC-6.4.0-2.28 impi/2017.3.196 Python/3.6.3

source /home/pperez/Thesis/sif2/venv/bin/activate

python exec.py /home/pperez/Thesis/sif2/inputs/hits15.csv /home/pperez/Thesis/sif2/inputs/dirset_leftraru.txt /home/pperez/Thesis/sif2/inputs/settings.txt
echo "Trabajo terminado";