#!/usr/bin/env sh
# Using pretrained weights
set -e

post_fix=msssimL2_10k_fine_tune_0.9

caffe train --solver=./solver_${post_fix}.prototxt --weights ~/Desktop/AOD-Net/AOD_Net.caffemodel 2>&1 | tee ../log/train_${post_fix}.log

python ~/Desktop/caffe/tools/extra/parse_log.py ../log/train_${post_fix}.log ../log/

python test.py -m ../data10k/solver_${post_fix}_iter_9000.caffemodel -o ../test/dehaze_${post_fix}