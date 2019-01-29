Implementation of our TIFS paper "WISERNet: Wider Separate-then-reunionNetwork for Steganalysis of Color Images"

Directories and files included in the implementation:

'model/wisernet_model.prototxt' - the model prototxt of WISERNet.

'solver/wisernet_solver.prototxt' - the solver prototxt of WISERNet as described in our paper.

'filler.hpp' - the filler implementation including SRM\kb\kv kernel. 

Please clone official Caffe framework from https://github.com/BVLC/caffe with commitment hashf19f0f17e711045d25dacb20f9b5ef6d39eb8aad, use filler.hpp provided by us to replace the official one in include/caffe/ directory and then re-compile Caffe.

'lmdb_create.py' - A python script to create lmdb training and testing file.

'wisernet.caffemodel' - The best performing WISERNet snapshot trained in BOSSBase PPG_LAN2 dataset with 0.4bpc HILL_CMD_C stego images.

'wisernet.sh' - Shell script to train and test the WISERNet model with corresponding solver and model prototxt.
