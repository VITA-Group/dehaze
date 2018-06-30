clc;
clear;
path_groundtruth_image = '../groundtruth';
path_dehazed_image = '../dehaze_1000samples_msssimL2_1800iters';

% get the file name of two images to be compared
listing = dir(path_dehazed_image);  %list all dehazing output
num_dehazed_image = length(listing) -2; % exclue . and .. returned by dir command
total_psnr = 0;
total_ssim = 0;
% num_dehazed_image = 10;

parfor i = 1: num_dehazed_image
    disp(i);
    if(listing(i).isdir == 0)
        dehazed_image = listing(i).name;
        image_idx = strtok(dehazed_image,'_');
        filename_groundtruth_image = fullfile(path_groundtruth_image,strcat(image_idx,'.png'));
        filename_dehazed_image = fullfile(path_dehazed_image,dehazed_image);
        groundtruth_image=imread(filename_groundtruth_image);
        dehazed_iamge=imread(filename_dehazed_image);
        
        % compute PSNR
        image_psnr = 0;
        for channel = 1:3
            x = groundtruth_image(:,:,channel);
            y = dehazed_iamge(:,:,channel);
            channel_psnr = psnr_633(x,y);
            image_psnr = image_psnr + channel_psnr;
        end
        
        % compute the SSIM
        image_ssim = ssim_633(dehazed_iamge, groundtruth_image);
        
        total_psnr = total_psnr + image_psnr/3;
        total_ssim = total_ssim + image_ssim;
    end
end

avg_psnr_dataset = total_psnr/num_dehazed_image
avg_ssim_dataset = total_ssim/num_dehazed_image

