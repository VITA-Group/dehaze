import numpy as np
import sys, os

# Edit the paths as needed:
caffe_root = '/scratch/user/nirajgoel/ml_project/fast-rcnn/new/caffe-resnet/faster-rcnn-resnet/caffe-fast-rcnn/'
#caffe_root = '../caffe/'
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, '/scratch/user/nirajgoel/ml_project/fast-rcnn/new/caffe-resnet/faster-rcnn-resnet/lib/')

import caffe


model = './models/bvlc_reference_caffenet/deploy.prototxt';
weights = './models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';

caffe.set_mode_cpu()

net = caffe.Net(model, weights, caffe.TEST)

for k in net.blobs:
    print k
print "PARAMS\n"
for k in net.params:
    print k
print "Layer names\n"

for k in net._layer_names:
    print k
