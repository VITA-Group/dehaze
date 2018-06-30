#!/usr/bin/env sh
set -e

post_fix=ssim_1k

caffe train --solver=./solver_$post_fix.prototxt 2>&1 | tee ../log/train_$post_fix.log

python ~/Desktop/caffe/tools/extra/parse_log.py ../log/train_$post_fix.log ../log/
