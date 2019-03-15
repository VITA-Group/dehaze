# Domain adaptation for MaskRCNN on haze image

## Description of the problem	

* The problem is important is because hazy environment happens frequently in the area with dense heavy industry. Thus, it's vital to have a model which can detect the objects efficiently in the hazy environment used in applications like public traffic monitor or automatic cars.

## Two dehaze ways
Two dehaze methods codes are provided in the corresponding folders. The readMEs contains in the folders. We still offer the implementation of AOD-Net in tensorflow, codes are in AOD_Tensorflow

## Major Challenge of the problem

* Limit size of haze dataset

## High light of our method

* Use domain adaptation for the MASK RCNN training.

## System requirement

* We use python 2.7 (python 3+ may have some small errors)
* cuDNN 7.0.5 cudatoolkit 8.0
* Tensorflow-gpu 1.5.0
* Keras
* pycocotools (if not installed you will get error when import coco configuration)

### Download requirement

* [Mask RCNN pre-trained model](https://github.com/matterport/Mask_RCNN/releases) or [DMASK RCNN1](https://drive.google.com/file/d/1l4pZtVQMRvu5seC_EewTN-A8SWk3KVwK/view?usp=sharing)
* [Haze image dataset](https://sites.google.com/view/reside-dehaze-datasets)
* [MSCC dehazed dataset](https://drive.google.com/file/d/1ZeA_WNYhVNVOc1bKQCQq-AZ2ZwscACax/view?usp=sharing) (Need to put in to the same folder with the Haze image)

If you want to train the model, you need to download the COCO dataset, the link can be found in the coco.py file. 

### Run examples

* Our main code is in the folder name "codes". To run "run_me.py" file, you need first download the Haze image dataset, and download the MSCC dataset, rename the MSCC dataset as "MSCCdehaze", then you need to download the Mask RCNN model, rename as "mask_rcnn_domain.h5". Then you can use command: `python run_me.py` to get the result.
* In folder AOD_NET and MSCNN dehaze, we show how to run the image dehaze process from the RTTS dataset, we also provided the dehazed image datasets.
