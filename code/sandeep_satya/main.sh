#!/bin/bash

module load CUDA/8.0.44-GCC-system
module load Anaconda/2-5.0.1
source activate faster-r-cnn

GPU_ID=0
cd rcnn
./experiments/scripts/test_faster_rcnn.sh $GPU_ID pascal_voc_0712 res101
