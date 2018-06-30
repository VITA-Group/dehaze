#!/usr/bin/env sh
set -e

caffe train --solver=./solver.prototxt 2>&1 | tee train_l2.log

python ~/Desktop/caffe/tools/extra/parse_log.py ./train_l2.log ./