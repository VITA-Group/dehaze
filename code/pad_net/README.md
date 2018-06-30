# PAD-Net: A Perception-Aided Single Image Dehazing Network
## Introduction
In this project, we investigate the possibility of replacing the L2 loss with perceptually derived loss functions (SSIM, MS-SSIM, etc.) in training an end-to-end dehazing neural network. Objective experimental results suggest that by merely changing the loss function we can obtain significantly higher PSNR and SSIM scores on the SOTS set in the RESIDE dataset, compared with a state-of-the-art end-to-end dehazing neural network (AOD-Net) that uses the L2 loss. The best PSNR we obtained was 23.50 (4.2% relative improvement), and the best SSIM we obtained was 0.8747 (2.3% relative improvement.) For more details, please read this [report](https://arxiv.org/pdf/1805.03146.pdf).

## System requirements
- Ubuntu 16.04 64 bit; other OSs were not tested
- [nvidia-caffe](https://github.com/NVIDIA/caffe) with CUDA 8 and cuDNN v7: branch `caffe-0.15`, commit `4b8d54d892116b9cb6822a917065a616f56b1292`; the original [BLVC caffe](https://github.com/BVLC/caffe) did not support the training scripts very well, but for testing purpose, the BVLC caffe should work
- PyCaffe has to be installed and included in your python search path. For example, run `export PYTHONPATH=$PATH_TO_CAFFE/python:$PYTHONPATH`, where `$PATH_TO_CAFFE` is the caffe root dir
- PyCaffe may need a lot of other dependencies, you can install anaconda to resolve most of them
- Matlab 2017a and up; older versions should work but not tested; if you don't have the parallel computing toolbox, just change all the `parfor` in `evaluate.m` to `for`
- Include `./src/psnr_633.m` and `./src/ssim_633.m` in your Matlab's search path

## Install
- Copy `loss.py` to `$PATH_TO_CAFFE/python/`
- Rename it to `pyloss.py`

## Test
- Open file `run.sh`
- Change `sots_dir` to where you put the RESIDE SOTS set. Please use absolute path. You can obtain this testing set [here](https://sites.google.com/view/reside-dehaze-datasets). We used the 1,000-image version of SOTS that contains 500 indoor and 500 outdoor images, and we assume that they are all put into `sots_dir`.
- Change `gt_dir` to where you put the groundtruth images. Please use absolute path. Again, we assume that you have put all the groundtruth images in the same folder
- Change `dehaze_dir` to the desired output directory. Please use absolute path.
- Run `./run.sh`
- The script may run for a while (<30 min) and may open your Matlab
- When it finishes, you can find the dehazed images in `dehaze_dir` and the average PSNR and SSIM printed on the screen. You can also find the PSNR and SSIM for each test sample in a `.mat` file named as `result_per_image_test.mat` under your current directory

## Pre-trained models
- data10k/solver_msssimL2_10k_fine_tune_iter_9000.caffemodel
    - PSNR: **23.43**
    - SSIM: **0.8747**
- data10k/solver_msssimL2_10k_fine_tune_0.7_iter_9000.caffemodel
    - PSNR: **23.50**
    - SSIM: **0.8676**

## Contact
- Guanlong Zhao (gzhao@tamu.edu), Department of Computer Science and Engineering, Texas A&M University
- Yu Liu (yliu129@tamu.edu), Department of Electrical and Computer Engineering, Texas A&M University
