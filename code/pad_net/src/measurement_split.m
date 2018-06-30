%% Compute PSNR and SSIM for indoor and outdoor separately
clc;
clear;
postfix = 'msssimL2_10k_fine_tune_0.7';
load(sprintf('../test/result_per_image_%s.mat', postfix),'measure_array');

%% Figure out indices
indoorIdx = 365:864;
outdoorIdx = [1:364, 865:1000];

%% Numbers
indoorPSNR = mean([measure_array(indoorIdx).PSNR])
indoorSSIM = mean([measure_array(indoorIdx).SSIM])

outdoorPSNR = mean([measure_array(outdoorIdx).PSNR])
outdoorSSIM = mean([measure_array(outdoorIdx).SSIM])