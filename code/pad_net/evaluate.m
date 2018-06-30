% clc;
% clear;
% postfix = 'msssimL2_10k_fine_tune';
% path_groundtruth_image = './test/groundtruth';
% path_dehazed_image = sprintf('./test/dehaze_%s', postfix);

% get the file name of two images to be compared
listing = dir(path_dehazed_image);  %list all dehazing output
num_dehazed_image = length(listing) -2; % exclue . and .. returned by dir command
listing = listing(3:end);
total_psnr = 0;
total_ssim = 0;
measure_array(num_dehazed_image).name = '';
measure_array(num_dehazed_image).PSNR = 0;
measure_array(num_dehazed_image).SSIM = 0;

tic
parfor i = 1:num_dehazed_image
    disp(i)
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
    image_psnr = image_psnr/3;
    image_ssim = ssim_633(dehazed_iamge, groundtruth_image);

    field1 = 'name';
    value1 = listing(i).name;
    field2 = 'PSNR';
    value2 = image_psnr;
    field3 = 'SSIM';
    value3 = image_ssim;
    st = struct(field1,value1,field2,value2,field3,value3);
    measure_array(i) = st;      

    total_psnr = total_psnr + image_psnr;
    total_ssim = total_ssim + image_ssim;
end

save('./result_per_image_test.mat','measure_array');
avg_psnr_dataset = total_psnr/num_dehazed_image;
fprintf('Average PSNR: %2.2f\n', avg_psnr_dataset);
avg_ssim_dataset = total_ssim/num_dehazed_image;
fprintf('Average SSIM: %1.4f\n', avg_ssim_dataset);
toc

