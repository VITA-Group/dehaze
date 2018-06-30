#!/bin/bash
echo "module load Anaconda/3-5.0.0.1"
module load Anaconda/3-5.0.0.1
echo "conda create -n tfdcp python=3.6.4"
conda create -n dcp-1 python=3.6.4
echo "source activate tf-dcp"
source activate dcp-1
echo "pip install cython easydict==1.6 numpy opencv-python pillow scipy matplotlib pyyaml"
pip install numpy==1.9.0 pillow==2.6.0
echo "conda install -c anaconda tensorflow-gpu"
#conda install -c anaconda tensorflow-gpu
echo "module load CUDA/9.1.85"
#module load CUDA/9.1.85
cd dark-channel-prior-dehazing
echo "python src/main.py -p RTTS"
python src/main.py -p RTTS
source deactivate
echo "now tf-faster-rcnn"
echo "cd lib/"
cd ../tf-faster-rcnn/lib/
conda create -n tf-1 python=3.6.4
echo "source activate tf-dcp"
source activate tf-1
echo "pip install cython easydict==1.6 numpy opencv-python pillow scipy matplotlib pyyaml"
pip install cython easydict==1.6 numpy opencv-python pillow scipy matplotlib pyyaml
echo "conda install -c anaconda tensorflow-gpu"
conda install -c anaconda tensorflow-gpu
echo "module load CUDA/9.1.85"
module load CUDA/9.1.85
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
ln -s ../../../../dark-channel-prior-dehazing/result JPEGImages
cd ../../..
echo "sbatch train.slurm"
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc_0712 res101
#sbatch train.slurm

