#!/usr/bin/env sh
set -e

post_fix=msssimL1_10k

caffe train --solver=./solver_$post_fix.prototxt 2>&1 -snapshot ../data10k/solver_msssimL1_10k_iter_5835.solverstate | tee ../log/train_$post_fix.log

python ~/Desktop/caffe/tools/extra/parse_log.py ../log/train_$post_fix.log ../log/
