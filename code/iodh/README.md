Improving Object Detection in Haze
------------------------------------
CSCE-633 Machine Learning - Project
------------------------------------
Authors: Niraj Goel, Ritu Raj
------------------------------------

USAGE
-------------------
Following settings are available to reproduce the results mentioned in the report

./tools/reval.py ./output/res101/voc_2007_test/default/res101_faster_rcnn_iter_110000/ --imdb voc_2007_test

- Run the evaluation script in already saved detections(on RTTS). This takes around 1 min.
    - steps:
        1. cd tf-faster-rcnn
        2. ./tools/reval.py ./output/res101/voc_2007_test/default/res101_faster_rcnn_iter_110000/ --imdb voc_2007_test
    - Notes:
        1. The dehazed images are present at: ./data/VOCdevkit2007/VOC2007/JPEGImages-dcp-refinedt
        2. The model is present at: ./output/res101/voc_2007_trainval+voc_2012_trainval/default/
        3. The saved detections are present inside: ./output/res101/voc_2007_test/default/
        4. RESIDE dataset has same structure as that of VOC, hence rather than updating the imdb name (voc_2007), we just 
           replace the voc images and annotations with RESIDE images and annotations respectively
        

- Run Object detection with the already provided dehazed image (with DCP) and trained faster R-CNN ResNet101 model. The setup takes
  approx 3-4 minutes and the detection on RTTS and evaluation takes around 30mins.
    - steps:
        1. cd tf-faster-rcnn
        2. source run_me.sh
    - Notes:
        1. The dehazed images are present at: ./data/VOCdevkit2007/VOC2007/JPEGImages-dcp-refinedt
        2. The model is present at: ./output/res101/voc_2007_trainval+voc_2012_trainval/default/

- Dehaze the image using DCP and run object detection with provided trained faster R-CNN ResNet101 model. This might take long time. 
  DCP takes more than 10hrs for dehazing 4322(RTTS) images sequentially.
    - steps:
        1. cd combined
        2. source run_me.sh
    - Notes:
        1. The dehazed images are generated inside: ./dark-channel-prior-dehazing/result 
        2. The model is present at: ./tf-faster-rcnn/output/res101/voc_2007_trainval+voc_2012_trainval/default/

RESULT LOGS
--------------------
The following files have result(logs) from our previous runs on various settings:
    - Object detection with RTTS original images   : ./result_logs/mAP_aod
    - Object detection with AOD-Net dehazed images : ./result_logs/mAP_aod
    - Object detection with DCP dehazed images     : ./result_logs/mAP_dcp

JOINT TUNING and DOMIAN ADAPTION
------------------------------------
The combined AOD-Net and Faster R-CNN model is present in the following directory
    - Network Prototxt files : ./caffe-resnet/faster-rcnn-resnet/models/aod_fast_rcnn
    - code to combine networks and copy weights: ./caffe-resnet/faster-rcnn-resnet/models/aod_fast_rcnn/combine.py  pretrained_temp1.caffemodel
    - combined model: ./caffe-resnet/faster-rcnn-resnet/models/aod_fast_rcnn/aodnet_fasterrcnn.caffemodel

The following directory contains domain adaption experiments, the Caffe build with domain adaption and Faster R-CNN layers. It also contains the 
combined ntwork definition files (Prototxt files)
    - Root directory: ./caffe-resnet/faster-rcnn-resnet
    - Network definition files: ./caffe-resnet/faster-rcnn-resnet/models/pascal_voc/ResNet101_BN_SCALE_Merged_domain_adap/faster_rcnn_end2end


Support
-------
- Please email your questions to:
    - nirajgoel@tamu.edu
    - rituraj131@tamu.edu
