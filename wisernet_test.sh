#!/usr/bin/env sh

../caffe/build/tools/caffe test -model wisernet_model_test.prototxt -weights wisernet.caffemodel -gpu 2 -iterations 10
