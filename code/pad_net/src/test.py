import os
import numpy as np
import caffe
import sys
from pylab import *
import re
import random
import time
import copy
import matplotlib.pyplot as plt
import cv2
import scipy
import shutil
import csv
from PIL import Image
import datetime
from sys import argv

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

def EditFcnProto(templateFile, height, width):
	with open(templateFile, 'r') as ft:
		template = ft.read()
        print templateFile
        outFile = 'DeployT.prototxt'
        with open(outFile, 'w') as fd:
            fd.write(template.format(height=height,width=width))

def run_test(input_path, output_path, model_path):
    caffe.set_mode_gpu()
    caffe.set_device(0)
    #caffe.set_mode_cpu();

    info = os.listdir(input_path)
    imagesnum=0
    for line in info:
        # reg = re.compile(r'(.*?).jpg')
        # all = reg.findall(line)
        if (line.endswith('png') or line.endswith('jpg')):
            imagename = line[0:-4]
            if (os.path.isfile(r'%s/%s' % (input_path, line)) == False):
                continue
            else:
                imagesnum = imagesnum + 1
                npstore = caffe.io.load_image('%s/%s' % (input_path, line))
                height = npstore.shape[0]
                width = npstore.shape[1]

                templateFile = 'test_template.prototxt'
                EditFcnProto(templateFile, height, width)

                model = model_path

                net = caffe.Net('DeployT.prototxt', model, caffe.TEST)
                batchdata = []
                data = npstore
                data = data.transpose((2, 0, 1))
                batchdata.append(data)
                net.blobs['data'].data[...] = batchdata

                net.forward()

                data = net.blobs['sum'].data[0]
                data = data.transpose((1, 2, 0))
                data = data[:, :, ::-1]

                savepath = output_path + '/' + imagename + '.jpg'
                cv2.imwrite(savepath, data * 255.0,[cv2.IMWRITE_JPEG_QUALITY, 100])

                print imagename

        print 'image numbers:',imagesnum

if __name__ == '__main__':
    # valid options
    #   -m: caffe model file path, mandatory
    #   -i: test img dir, optional, default: "../test/input"
    #   -o: output dir path, optional, default: "../test/output"
    # Example: python test.py -m ../toy_data/final.caffemodel -i ../test/input -o ../test/output

    myargs = getopts(argv)

    model_path = ''
    input_path = '../test/input'
    output_path = '../test/output'
    isRun = True

    if '-m' in myargs:
        model_path = myargs['-m']
        if not os.path.isfile(model_path):
            isRun = False
            print 'Please check model path, use absolute path'

    if '-i' in myargs:
        input_path = myargs['-i']
        if not os.path.exists(input_path):
            isRun = False
            print 'Please check input path, use absolute path'
    
    if '-o' in myargs:
        output_path = myargs['-o']
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        else:
            # delete previous run
            for the_file in os.listdir(output_path):
                file_path = os.path.join(output_path, the_file)
                try:
                    if os.path.isfile(file_path): os.remove(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)

    if isRun:
        run_test(input_path, output_path, model_path)
    else:
        print 'Please check your input parameters!'