import time
from model import *
from utils import*
import DMaskRCNN as model
import coco
from extract import parse_annotation, resize_bbox
import skimage.io


# Root directory of the project
root_dir = os.getcwd()

# Directory to save logs and trained model
model_dir = os.path.join(root_dir, "domain_logs")

# Local path to trained weights file
dehaze_model_path = os.path.join(root_dir, "mask_rcnn_domain.h5")
best_mAP_model_path = os.path.join(root_dir, "best_mAP_maskRCNN.h5")

test_image_dir = os.path.join(root_dir, "RTTS/JPEGImages")
test_file_dir = os.path.join(root_dir, "RTTS/Annotations")
test_image_names = next(os.walk(test_image_dir))[2]
test_image_names.sort()

domain_image_dir = os.path.join(root_dir, "HazeTrain")
domain_image_names = next(os.walk(domain_image_dir))[2]
domain_image_names.sort()

COCO_DIR = "COCO"
config = coco.CocoConfig()
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
# config.display()

# Get the first available GPU
DEVICE_ID_LIST = GPUtil.getFirstAvailable()
DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

# Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

# Training dataset. Use the training set and 35K from the
# validation set, as as in the Mask RCNN paper.

year = 2014
dataset_train = coco.CocoDataset()
dataset_train.load_coco(COCO_DIR, "train", year=year, auto_download=True)
dataset_train.load_coco(COCO_DIR, "valminusminival", year=year, auto_download=True)
dataset_train.prepare()


def mAP_test(image_names, image_dir, file_dir):
    APs = []
    t1 = time.time()
    for image_name in image_names:
        image = skimage.io.imread(os.path.join(image_dir, image_name))
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        gt_class_id, ori_gt_bbox = parse_annotation(image_name, file_dir)
        gt_bbox = resize_bbox(ori_gt_bbox, scale, padding)
        results = domain_model.detect([image], verbose=0)
        r = results[0]
        AP, precisions, recalls, overlaps =\
            utils.compute_box_ap(gt_bbox, gt_class_id,
                             r["rois"], r["class_ids"], r["scores"])
        APs.append(AP)
    t2 = time.time()-t1
    print ("using time: ", t2)
    print("mAP: ", np.mean(APs))
    return np.mean(APs)

## Load domain model
domain_model = model.Domain_MaskRCNN(mode="training", model_dir=model_dir,
                          config=config)

# Or load the last model you trained
# weights_path = model.find_last()[1]
domain_model.load_weights(dehaze_model_path, by_name=True)

# Training - Stage 1
print("Training domain classification model")
domain_model.train(dataset_train, dataset_train,  domain_image_dir, domain_image_names,
            learning_rate=config.LEARNING_RATE,
            epochs=5,
            layers='domain')

domain_model.keras_model.save_weights(dehaze_model_path )

del domain_model

mAP = 0

for i in range(5):

    print('Training all layer at epoch', i)
    domain_model = model.Domain_MaskRCNN(mode="training", model_dir=model_dir,
                                         config=config)

    domain_model.load_weights(dehaze_model_path, by_name=True)

    print("Fine tune all layers")
    domain_model.train(dataset_train, dataset_train, domain_image_dir, domain_image_names,
                learning_rate=config.LEARNING_RATE ,
                epochs=10,
                layers='all')
    domain_model.keras_model.save_weights(dehaze_model_path)


    del domain_model

    domain_model = model.Domain_MaskRCNN(mode="inference", model_dir=model_dir,
                          config=config)
    domain_model.load_weights(dehaze_model_path, by_name=True)

    cur_mAP =  mAP_test(test_image_names, test_image_dir, test_file_dir)
    if cur_mAP>mAP:
        mAP = cur_mAP
        print ('find currently best model, mAP=', mAP)
        domain_model.keras_model.save_weights(best_mAP_model_path)
        print('model saved in path: ', best_mAP_model_path)

    del domain_model

for i in range(2):

    print('Fine tune all layer at epoch', i)
    domain_model = model.Domain_MaskRCNN(mode="training", model_dir=model_dir,
                                         config=config)

    domain_model.load_weights(dehaze_model_path, by_name=True)

    print("Fine tune all layers")
    domain_model.train(dataset_train, dataset_train, domain_image_dir, domain_image_names,
                       learning_rate=config.LEARNING_RATE/10,
                       epochs=10,
                       layers='all')
    domain_model.keras_model.save_weights(dehaze_model_path)

    del domain_model

    domain_model = model.Domain_MaskRCNN(mode="inference", model_dir=model_dir,
                                         config=config)
    domain_model.load_weights(dehaze_model_path, by_name=True)

    cur_mAP = mAP_test(test_image_names, test_image_dir, test_file_dir)
    if cur_mAP>mAP:
        mAP = cur_mAP
        print ('find currently best model, mAP=', mAP)
        domain_model.keras_model.save_weights(best_mAP_model_path)
        print('model saved in path: ', best_mAP_model_path)

    del domain_model