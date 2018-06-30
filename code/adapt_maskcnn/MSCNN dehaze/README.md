
# Description
We use this codes to process our dehazing process.
This orginal codes are a test implementation for ECCV16 paper: Single Image Dehazing via Multi-scale Convolutional Neural Networks. Wenqi Ren, Si Liu, etc. It provides the pretrained model.  
We modifies the orginal codes to process all the images we have as a whole process. The modified codes we named it "MSCNNdehazing"

# Prepare
This dehaze code is based on the MatConvNet toolbox.  You should compile MatConvNet first on your computer

# Generate dehazed pictures

To generate dehazed pictures, first make a folder named "imgs" and put all pictures we want to dehaze in this folder. After that, run the matlab code "MSCNNdehazing.m". After finised, the dehazed pictures will be stored in the folder named "result".

