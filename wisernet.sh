#!/usr/bin/env sh

../caffe/build/tools/caffe train -solver ./solver/wisernet_solver.prototxt -gpu 1 2>&1 | tee  ./log/wisernet_exp.log  

#../caffe/build/tools/caffe test -model model/wisernet_model.prototxt -weights wisernet.caffemodel -gpu 2 -iterations 2000
