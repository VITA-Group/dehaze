import  numpy as np
import os
from xml.dom import minidom

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorbike', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def get_name_id(name):
    return class_names.index(name)


def parse_annotation(image_name, file_dir):
    '''
    return lists of [name, y1, x1, y2, x2]
    file_dir is the annotation dir
    '''
    name = image_name.split('.')[0]
    file_name = name + '.xml'
    xml_file = os.path.join(file_dir, file_name)
    parse_doc = minidom.parse(xml_file)
    objects = parse_doc.getElementsByTagName("object")
    parse_metas = []
    for file_object in objects:
        parse_meta = []
        name = file_object.getElementsByTagName("name")[0]
        name_id = get_name_id(str(name.firstChild.data))
        parse_meta.append(name_id)
        xmax = file_object.getElementsByTagName("xmax")[0]
        ymax = file_object.getElementsByTagName("ymax")[0]
        xmin = file_object.getElementsByTagName("xmin")[0]
        ymin = file_object.getElementsByTagName("ymin")[0]
        parse_meta.append(int(ymin.firstChild.data))
        parse_meta.append(int(xmin.firstChild.data))
        parse_meta.append(int(ymax.firstChild.data))
        parse_meta.append(int(xmax.firstChild.data))
        parse_metas.append(parse_meta)
    gt_class_id = []
    gt_bbox = []
    for i in parse_metas:
        gt_class_id.append(i[0])
        gt_bbox.append(i[1:])
    gt_class_id = np.array(gt_class_id)
    gt_bbox = np.array(gt_bbox)
    return gt_class_id, gt_bbox

def resize_bbox(bbox, scale, padding):
    new_bbox = []
    for box in bbox:
        box = scale*box
        # y1
        box[0] = box[0] + padding[0][1]
        # x1
        box[1] = box[1] + padding[1][1]
        # y2
        box[2] = box[2] + padding[0][1]
        # y2
        box[3] = box[3] + padding[1][1]
        new_bbox.append(box)
    new_bbox = np.array(new_bbox)
    return new_bbox