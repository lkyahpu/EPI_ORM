#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:21:21 2019

@author: lky
"""

from __future__ import print_function

import keras

import matplotlib.pyplot as plt

import numpy as np

import os

import imageio
from read_LF import read_all_LF ,read_pinhole_LF

from keras.models import load_model



    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"    

 
model_512=load_model('E:/LKY/paper/EPI_paper/My paper/EPI_ORM/EPI_test/model.hdf5')
    
    
#valdata_fill_x ,valdata_fill_y ,numy,numx= read_all_LF(9,9,'buddha.png')     #,valdata_fill_xd ,valdata_fill_yd
valdata_fill_x ,valdata_fill_y ,numy,numx= read_pinhole_LF(9,9,'E:/LKY/dataset/benchmark/cotton')   #,valdata_fill_xd ,valdata_fill_yd
   
         
_,_,vald_output=model_512.predict([valdata_fill_x,valdata_fill_y])
    
vald_output=vald_output.reshape(numy,numx,1)
    
plt.imshow(vald_output[:,:,0])
#plt.imsave('occlusion.png',vald_output[:,:,0])
#vald_output=(((vald_output+439)/1149)*255).astype(np.uint8) 

imageio.imsave('E:/LKY/paper/EPI_paper/My paper/EPI_ORM/EPI_test/cotton.png',vald_output)
print('finished')
