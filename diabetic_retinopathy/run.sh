#!/bin/bash -l 
 
# Slurm parameters 
<<<<<<< HEAD
#SBATCH --job-name=Job_Name_13 
=======
#SBATCH --job-name=train 
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
#SBATCH --output=job_name-%j.%N.out 
#SBATCH --time=1-00:00:00 
#SBATCH --gpus=1 
 
# Activate everything you need 
module load cuda/11.2 
# Run your python code 
python3 main.py
