#!/usr/bin/env sh
# Sample testing code using a pre-trained model
set -e

# CHANGE THIS
# The path to your SOTS set; we assume that you put the indoor and outdoor images in the same folder
# Please use absolute path
sots_dir="./test/input/"

# CHANGE THIS
# The path to the groundtruth images; we assume that you put the indoor and outdoor images in the same folder
# Please use absolute path
gt_dir="./test/groundtruth/"

# CHANGE THIS
# The path to the output folder
# Please use absolute path
dehaze_dir="./test/dehaze_test/"

# Generating dehazed images using a pre-trained model (MSSSIM+L2 (alpha=0.1))
python src/test.py -m ./data10k/solver_msssimL2_10k_fine_tune_iter_9000.caffemodel -i $sots_dir -o $dehaze_dir

# Run evaluation and output PSNR and SSIM
# Remember to include src/psnr_633.m and src/ssim_633.m in your Matlab's search path
# You will find a .mat file named "result_per_image_test.mat" under your current dir, which contains PSNR and SSIM for each test image
matlab_commands="addpath('./src/');path_groundtruth_image='$gt_dir';path_dehazed_image='$dehaze_dir';evaluate;"
matlab -r $matlab_commands