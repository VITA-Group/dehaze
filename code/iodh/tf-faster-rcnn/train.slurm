#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
##SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L              #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=train             #Set the job name to "JobExample4"
#SBATCH --time=00:48:00
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=10G                  #Request 2560MB (2.5GB) per node
#SBATCH --output=train_log.%j	   #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=122822455777             #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=rituraj131@tamu.edu    #Send all emails to email_address

#First Executable Line


echo $PATH
module list

nvidia-smi

source activate tf-dcp
module load CUDA/9.1.85
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc_0712 res101

