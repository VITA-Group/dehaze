#!/usr/bin/env sh
set -e

caffe train --solver=./solver_msssimL2.prototxt $@
