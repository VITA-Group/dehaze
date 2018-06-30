#!/bin/bash
echo "module load Anaconda/3-5.0.0.1"
module load Anaconda/3-5.0.0.1
echo "conda create -n tfdcp python=3.6.4"
conda create -n tf-dcp python=3.6.4
echo "source activate tf-dcp"
source activate tf-dcp
echo "pip install cython easydict==1.6 numpy opencv-python pillow scipy matplotlib pyyaml"
pip install cython easydict==1.6 numpy opencv-python pillow scipy matplotlib pyyaml
echo "conda install -c anaconda tensorflow-gpu"
conda install -c anaconda tensorflow-gpu
echo "module load CUDA/9.1.85"
module load CUDA/9.1.85
echo "cd lib/"
cd lib/
echo "make clean"
make clean
echo "make"
make
cd ..
echo "cd data/coco/PythonAPI"
cd data/coco/PythonAPI
echo "make"
make
cd ../../..
echo "cd data/VOCdevkit2007/VOC2007"
cd data/VOCdevkit2007/VOC2007
rm -r JPEGImages
echo "ln -s JPEGImages-dcp-refinedt JPEGImages"
ln -s JPEGImages-dcp-refinedt JPEGImages
cd ../../..
echo "sbatch train.slurm"
sbatch train.slurm


