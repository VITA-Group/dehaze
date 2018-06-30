#!/usr/bin/env sh
set -e

caffe train --solver=./solver_msssim.prototxt $@
