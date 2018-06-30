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
model_path= os.path.join(root_dir, "mask_rcnn_domain.h5")

test_image_dir = os.path.join(root_dir, "RTTS/JPEGImages")
test_aod_dir = os.path.join(root_dir, "RTTS/AOD_Dehaze")
test_MSCC_dir = os.path.join(root_dir, "RTTS/MSCCdehaze")
test_file_dir = os.path.join(root_dir, "RTTS/Annotations")
test_image_names = next(os.walk(test_image_dir))[2]
test_image_names.sort()


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



def mAP_test(image_dir, file_dir):
    APs = []
    t1 = time.time()
    image_names = next(os.walk(image_dir))[2]
    image_names.sort()
    for image_name in image_names:
        image = skimage.io.imread(os.path.join(image_dir, image_name))
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        if image_name.find('AOD') != -1:
            image_name = image_name.replace('_AOD-Net', '')
        if image_name.find('dehaze') != -1:
            image_name = image_name.replace('_dehazed', '')
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


print('Build the mask rcnn model')
domain_model = model.Domain_MaskRCNN(mode="inference", model_dir=model_dir,
                                 config=config)
print('loading weights from: ', model_path)
domain_model.load_weights(model_path, by_name=True)


print('calculating the mAP from dehazed MSCC dataset')
cur_mAP2 = mAP_test(test_MSCC_dir  , test_file_dir)
print('current mAP is:', cur_mAP2)

del domain_model
K.clear_session()

