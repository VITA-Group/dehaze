#!/bin/bash
# valid options
#   -t: num of training samples, default to 1000
#   -v: num of val samples, default to 1000
# Example: python create_data_list.py -t 10000 -v 1000
# 
# This script will first find all possible haze images, and then shuffle their order
# and then pick the first val_size files as the val set, and then pick the following
# train_size files as the training set
# The resulting list for training and val will reflect the distribution of OTS
# and ITS in RESIDE, so we do not have equal number of images from two groups

from glob import glob
import os
import math
import random
from sys import argv

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

def getCleanName(in_name):
    return in_name.split('.')[0].split('_')[0]

## get input params
myargs = getopts(argv)
train_size = 1000
val_size = 1000
if '-t' in myargs:
    train_size = int(myargs['-t'])
if '-v' in myargs:
    val_size = int(myargs['-v'])
print 'Num train imgs: %s\nNum val imgs: %d' % (train_size, val_size)

## global settings
fake_label = 0 # we do not care the label's value
its_val_start = 10001 # its val set starts from 10001.png
data_post_fix = '%dk' % (round(train_size/1000))

## setup dirs
raw_data_root = '/media/guanlong/DATA/633_data/'
its_train_haze = raw_data_root + 'ITS/train/ITS_haze/'
its_train_clean = raw_data_root + 'ITS/train/ITS_clear/'
its_val_haze = raw_data_root + 'ITS/val/haze/'
its_val_clean = raw_data_root + 'ITS/val/clear/'
ots_haze = raw_data_root + 'OTS_haze/'
ots_clean = raw_data_root + 'OTS_clean/'
data_root = '../data%s' % (data_post_fix)
data_train = '%s/train/' % (data_root)
data_val = '%s/val/' % (data_root)
if not os.path.exists(data_root):
    os.mkdir(data_root)
if not os.path.exists(data_train):
    os.mkdir(data_train)
if not os.path.exists(data_val):
    os.mkdir(data_val)
train_haze_txt = open('%s/train_haze.txt' % (data_root), 'w')
train_clean_txt = open('%s/train_clean.txt' % (data_root), 'w')
val_haze_txt = open('%s/val_haze.txt' % (data_root), 'w')
val_clean_txt = open('%s/val_clean.txt' % (data_root), 'w')

## get full file list
its_haze_list = (glob(its_train_haze + '*.png') + glob(its_val_haze + '*.png'))
its_haze_list.sort()
its_clean_list = [] # get the clean file for each haze image
for haze_file in its_haze_list:
    haze_name = haze_file.split('/')[-1]
    clean_idx = getCleanName(haze_name)
    clean_idx_num = int(clean_idx)
    clean_name = clean_idx + '.png'
    isInTrainSet = (clean_idx_num < its_val_start)
    if isInTrainSet: # some one did a stupid data split
        clean_file = '%s%s' % (its_train_clean, clean_name)
    else:
        clean_file = '%s%s' % (its_val_clean, clean_name)
    its_clean_list.append(clean_file)

ots_haze_list = glob(ots_haze + '*.jpg')
ots_haze_list.sort()
ots_clean_list = [] # get the clean file for each haze image
for haze_file in ots_haze_list:
    haze_name = haze_file.split('/')[-1]
    clean_idx = getCleanName(haze_name)
    clean_name = clean_idx + '.jpg'
    clean_file = '%s%s' % (ots_clean, clean_name)
    ots_clean_list.append(clean_file)

haze_list = its_haze_list + ots_haze_list
clean_list = its_clean_list + ots_clean_list
# print 'Num haze image: %d\nNum clean image: %d' % (len(haze_list), len(clean_list))
assert len(haze_list) == len(clean_list)

# shuffle the order of files
combined = list(zip(haze_list, clean_list))
random.shuffle(combined)
haze_list[:], clean_list[:] = zip(*combined)
numImgs = len(haze_list)

## build the txt files
# first fulfill val requirement
if numImgs > val_size: # enough for val, not equal to ensure at least one training
    val_list = range(0, val_size)
    if (numImgs - val_size) >= train_size: # enough for training
        train_list = range(val_size, (val_size+train_size))
    else: # not enough for train, use the rest
        train_list = range(val_size, numImgs)
        print 'No enough OTS samples for training, actual training set is %d images' %(len(train_list))
else: # not enough for val
    print 'You do not need such a big val set from ITS, go pick a smaller one!'
    exit()

# get the txt files
for ii in val_list:
    haze_output = '%s %d\n' % (haze_list[ii], fake_label)
    val_haze_txt.write(haze_output)
    clean_output = '%s %d\n' % (clean_list[ii], fake_label)
    val_clean_txt.write(clean_output)

for jj in train_list:
    haze_output = '%s %d\n' % (haze_list[jj], fake_label)
    train_haze_txt.write(haze_output)
    clean_output = '%s %d\n' % (clean_list[jj], fake_label)
    train_clean_txt.write(clean_output)

train_haze_txt.close()
train_clean_txt.close()
val_haze_txt.close()
val_clean_txt.close()