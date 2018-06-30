#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
##SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L              #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=train             #Set the job name to "JobExample4"
#SBATCH --time=1-01:30:00
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=5120M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=train_log.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue
##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=nirajgoel@tamu.edu    #Send all emails to email_address 

#First Executable Line


echo $PATH
module list

nvidia-smi


module load CUDA/9.0.176 ; module load cuDNN/7.0.5-CUDA-9.0.176 ; module load OpenCV/3.4.1-foss-2017b-Python-2.7.14-CUDA-9.0.176 ;module load Cython/0.25.2-GCCcore-6.3.0-Python-2.7.12-bare ; module load GCCcore/4.9.3 ; source activate fast-rcnn1-copy ; export PYTHONPATH= ; export LD_LIBRARY_PATH=/scratch/user/nirajgoel/anaconda3/envs/fast-rcnn1/lib/:$LD_LIBRARY_PATH

./tools/test_net.py --gpu 0 --def models/aod_fast_rcnn/test.prototxt --net /scratch/user/nirajgoel/ml_project/fast-rcnn/new/caffe-resnet/faster-rcnn-resnet/models/aod_fast_rcnn/pretrained_temp.caffemodel --imdb voc_0712_test --cfg experiments/cfgs/faster_rcnn_end2end.yml |&tee log_test_temp
