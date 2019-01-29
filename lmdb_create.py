#!/usr/bin/python
# -*- coding: utf-8 -*-
#The program generate the lmdb dataset for the Caffe input
#Implement in python-2.7
#Author:JiShen Zeng 2015/12/08
#Input: Image data directory with train set directory and test set directory
#Output: lmdb file for caffe 

import caffe
import lmdb
from scipy import ndimage
import os
import numpy as np
from caffe.proto import caffe_pb2

#prepare the image dir and train test lists                                                
cover_path="/data/BOSS_PPG_LAN2/"
stego_path="/data/HILL_CMDC/BOSS_PPG_LAN2_HILLCMD_40/"

#train                                                                                
train_image_names_string=os.popen("cat train_boss.list").read()
train_image_names=train_image_names_string.split('\n')[0:-1]

#test                                                                                
test_image_names_string=os.popen("cat test_boss.list").read()
test_image_names=test_image_names_string.split('\n')[0:-1]


#basic setting
lmdb_file = 'data/boss_ppg_lan2_hillcmdc_40_test'
batch_size = 6400

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file, map_size=int(1e13))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

item_id = -1
image_id= 0
for x in range(10000):
    item_id += 1
    
    #prepare the data and label
    if(item_id % 2) == 0:
        image_path=os.path.join(cover_path,test_image_names[image_id])
        images=ndimage.imread(image_path)
        images=images.transpose((2,0,1))
        data=images
        label=1
    else :
        image_path=os.path.join(stego_path,test_image_names[image_id])
        images=ndimage.imread(image_path)
        images=images.transpose((2,0,1))
        data=images
        label=0
        image_id+=1

    # save in datum
    datum = caffe.io.array_to_datum(data, label)
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    # write batch
    if(item_id + 1) % batch_size == 0:
        lmdb_txn.commit()
        lmdb_txn = lmdb_env.begin(write=True)
        print (item_id + 1)

# write last batch
if (item_id+1) % batch_size != 0:
    lmdb_txn.commit()
    print 'last batch'
    print (item_id + 1)


#basic setting
lmdb_file = 'data/boss_ppg_lan2_hillcmdc_40_train'
batch_size = 6400

# create the lmdb file
lmdb_env = lmdb.open(lmdb_file, map_size=int(1e13))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe_pb2.Datum()

item_id = -1
image_id= 0
for x in range(10000):
    item_id += 1
    
    #prepare the data and label
    if(item_id % 2) == 0:
        image_path=os.path.join(cover_path,train_image_names[image_id])
        images=ndimage.imread(image_path)
        images=images.transpose((2,0,1))
        data=images
        label=1
    else :
        image_path=os.path.join(stego_path,train_image_names[image_id])
        images=ndimage.imread(image_path)
        images=images.transpose((2,0,1))
        data=images
        label=0
        image_id+=1

    # save in datum
    datum = caffe.io.array_to_datum(data, label)
    keystr = '{:0>8d}'.format(item_id)
    lmdb_txn.put( keystr, datum.SerializeToString() )

    # write batch
    if(item_id + 1) % batch_size == 0:
        lmdb_txn.commit()
        lmdb_txn = lmdb_env.begin(write=True)
        print (item_id + 1)

# write last batch
if (item_id+1) % batch_size != 0:
    lmdb_txn.commit()
    print 'last batch'
    print (item_id + 1)


