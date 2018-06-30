#!/usr/bin/env sh
# Create the reside lmdb inputs
set -e

OUTPUT=../data10k
DATA=../data10k
TOOLS=../../caffe/build/tools

# Remove previous run
# rm -r $OUTPUT/train/reside_train_clean_lmdb
# rm -r $OUTPUT/train/reside_train_haze_lmdb
# rm -r $OUTPUT/val/reside_val_clean_lmdb
# rm -r $OUTPUT/val/reside_val_haze_lmdb

# Set RESIZE=true to resize the images. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=641
  RESIZE_WIDTH=641
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo "Creating train haze lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    "" \
    $DATA/train_haze.txt \
    $OUTPUT/train/reside_train_haze_lmdb

echo "Creating train clean lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    "" \
    $DATA/train_clean.txt \
    $OUTPUT/train/reside_train_clean_lmdb

echo "Creating val haze lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    "" \
    $DATA/val_haze.txt \
    $OUTPUT/val/reside_val_haze_lmdb

echo "Creating val clean lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    "" \
    $DATA/val_clean.txt \
    $OUTPUT/val/reside_val_clean_lmdb

echo "Done."
