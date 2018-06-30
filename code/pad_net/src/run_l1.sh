#!/usr/bin/env sh
set -e

caffe train --solver=./solver_l1.prototxt $@
