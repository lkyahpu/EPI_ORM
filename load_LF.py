#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:32:01 2019

@author: lky
"""
import os
import numpy as np
from file_io2 import read_lightfield
from file_io2 import read_pfm
from keras.utils import to_categorical
from imageio import imread
#from image_augumentation import randomColor,randomGaussian
#from PIL import Image


def load_LFdata(dir_LFimages):    
       #traindata_all=np.zeros((len(dir_LFimages), 9, 9, 512, 512, 3),np.uint16)
       traindata_y=np.zeros((len(dir_LFimages),9,512,512,3),np.float32)
       traindata_x=np.zeros((len(dir_LFimages),9,512,512,3),np.float32)            
       traindata_label=np.zeros((len(dir_LFimages), 512, 512),np.float32)

       v_s=['004','013','022','031','040','049','058','067','076']
       h_s=['036','037','038','039','040','041','042','043','044']

       #batch_size=len(dir_LFimages)
       image_id=0

       for dir_LFimage in dir_LFimages:
          n=0
          for vs in v_s:
            traindata_y[image_id,n,:,:,:]=imread(dir_LFimage+'/input_Cam'+vs+'.png').astype('float32')
            n=n+1
          n=0
          for hs in h_s:
            traindata_x[image_id,n,:,:,:]=imread(dir_LFimage+'/input_Cam'+hs+'.png').astype('float32')
            n=n+1  
         
          traindata_label[image_id,:,:]=np.float32(read_pfm(dir_LFimage+'/gt_disp_lowres.pfm')).astype('float32')  
        
          image_id=image_id+1

       '''
       traindata_x=traindata_x.transpose(0,3,2,1,4)
       traindata_x=traindata_x.transpose(0,2,1,3,4)
       traindata_x=traindata_x.reshape(len(dir_LFimages),512,512*9,3)
         
        
       
       traindata_y=traindata_y.transpose(0,2,1,3,4)
       traindata_y=traindata_y.reshape(len(dir_LFimages),512*9,512,3)
       '''
       #traindata_label = traindata_label[:,:,:,np.newaxis]
        
       return traindata_x, traindata_y, traindata_label    #,len(dir_LFimages) ,traindata_x_new, traindata_y_new
     

def generate_traindata(traindata_x,traindata_y, traindata_label,batch_size):  #, traindata_d_1,traindata_d_2
    
    
    input_size1=9
    input_size2=29
    input_size=29
    traindata_batch_x=np.zeros((batch_size,input_size1,input_size2,3),dtype=np.float32)
    traindata_batch_y=np.zeros((batch_size,input_size1,input_size2,3),dtype=np.float32)
    #traindata_batch_y_tmp=np.zeros((batch_size,input_size2,input_size1,3),dtype=np.float32)
    
    traindata_label_batchNxN=np.zeros((batch_size,1,1),dtype=np.float32)
    

    '''
    
    #data augmentation refocusing
    cat = np.concatenate
    w = input_size1
    h = input_size1
    hw = int(w / 2)
    hh = int(h / 2)

    disp_range =[x for x in range(-3,4,1)]
    disp=np.random.choice(disp_range)
    
    for i in range(w):
      shift = disp * (i - hw)
      traindata_x[:, i, :, :, :] = cat([traindata_x[:, i, :, -shift:,:],traindata_x[:, i, :, :-shift,:]], -2)

    
    for i in range(h):
      shift = disp * (i - hh)
      traindata_y[:, i, :, :, :] = cat([traindata_y[:, i, -shift:, :, :],traindata_y[:, i, :-shift, :, :]], -3)

    # correct ground truth
        
    traindata_label -= float(disp)
    '''
    

    for ii in range(0,batch_size):
        
        aa_arr =[x for x in range(0,16)]     
 
        image_id=np.random.choice(aa_arr)
        
        #image_id = ii
        
        aa_arr1 =[x for x in range(0,484)]
        id_start1 = np.random.choice(aa_arr1)
        aa_arr2 =[x for x in range(0,484)]
        id_start2 = np.random.choice(aa_arr2)
        
        
            
        traindata_batch_x[ii,:,:,:]= traindata_x[image_id:image_id+1, :,(id_start1+14):(id_start1+14)+1, (id_start2+14)-(input_size//2):(id_start2+14)+(input_size//2)+1,:].reshape(9,29,3)
                                               #G*traindata_x[image_id:image_id+1, id_start1+14-(input_size2//2):id_start1+14+(input_size2//2)+1, (id_start2+14)*9+4-(input_size1//2):(id_start2+14)*9+4+(input_size1//2)+1, 1].astype('float32')+
                                               #B*traindata_x[image_id:image_id+1, id_start1+14-(input_size2//2):id_start1+14+(input_size2//2)+1, (id_start2+14)*9+4-(input_size1//2):(id_start2+14)*9+4+(input_size1//2)+1, 2].astype('float32'))[:,:,np.newaxis]
        
        traindata_batch_y[ii,:,:,:]= traindata_y[image_id:image_id+1, :,(id_start1+14)-(input_size//2):(id_start1+14)+(input_size//2)+1, (id_start2+14):(id_start2+14)+1,:].reshape(9,29,3)
                                               #G*traindata_y[image_id:image_id+1, (id_start1+14)*9+4-(input_size1//2):(id_start1+14)*9+4+(input_size1//2)+1, id_start2+14-(input_size2//2):id_start2+14+(input_size2//2)+1, 1].astype('float32')+
        
        #traindata_batch_y[ii,:,:,:]= np.copy(np.rot90(traindata_batch_y_tmp[ii,:,:,:],1,(0,1)))

        
        
        traindata_label_batchNxN[ii,:,:]=traindata_label[image_id ,id_start1+14:id_start1+14+1,id_start2+14:id_start2+14+1].astype('float32')
        
    
    traindata_batch_x=np.float32((1/255)*traindata_batch_x)
    traindata_batch_y=np.float32((1/255)*traindata_batch_y)
    
    traindata_batch_x=np.minimum(np.maximum(traindata_batch_x,0),1)
    traindata_batch_y=np.minimum(np.maximum(traindata_batch_y,0),1)
    
    return traindata_batch_x,traindata_batch_y,traindata_label_batchNxN #,traindata_batch_d_1,traindata_batch_d_2





def generate_valdata(dir_LFimages):            
    
    
    input_size=29
    
   
    num_x=482
    num_y=482
    num=0
    
    #valdata_batch_x=np.zeros((1*num_x*num_y,input_size1,input_size2,3),dtype=np.float32)
    #valdata_batch_y=np.zeros((1*num_x*num_y,input_size1,input_size2,3),dtype=np.float32)
    valdata_batch_x=np.zeros((1*num_x*num_y,9,input_size,3),dtype=np.float32)
    valdata_batch_y=np.zeros((1*num_x*num_y,9,input_size,3),dtype=np.float32)

#valdata_batch_x_1=np.zeros((1*num_x*num_y,15,15,7),dtype=np.float32)
#valdata_batch_y_1=np.zeros((1*num_x*num_y,15,15,7),dtype=np.float32)
    
    valdata_label_batch_482=np.zeros((num_x*num_y,1,1),dtype=np.float32)    

    valdata_x, valdata_y, valdata_label = load_LFdata(dir_LFimages)
    
    #valdata_fill_y=valdata_fill_y[:,:,:,np.newaxis]

    
    for kk in range(0,num_x,1):
     for n in range(0,1):
      for jj in range(0,num_y,1):     #valdata_batch_y[num,:,:,:]=valdata_y[:, kk*9: kk*9+9, jj: jj+19, :].astype('float32')
       valdata_batch_x[num,:,:,:]=valdata_x[n,:, (kk+15) : (kk+15)+1, (jj+15)-(input_size//2):(jj+15)+(input_size//2)+1, :].reshape(9,29,3)
       valdata_batch_y[num,:,:,:]=valdata_y[n,:, (kk+15)-(input_size//2) : (kk+15)+(input_size//2)+1, (jj+15):(jj+15)+1, :].reshape(9,29,3)     
       valdata_label_batch_482[num,:,:]=valdata_label[n,kk+15:kk+15+1,jj+15:jj+15+1]
    
       num=num+1
      
     
          
    valdata_batch_x=np.float32((1/255)*valdata_batch_x)
    valdata_batch_y=np.float32((1/255)*valdata_batch_y)

    
    valdata_batch_x=np.minimum(np.maximum(valdata_batch_x,0),1)
    valdata_batch_y=np.minimum(np.maximum(valdata_batch_y,0),1)    
    
    
    
    return valdata_batch_x, valdata_batch_y , valdata_label_batch_482   #,valdata_batch_d_1,valdata_batch_d_2
    
   

def data_augmentation_for_train(traindata_batch_x,traindata_batch_y,batch_size):
      for batch_i in range(int(batch_size)):
        
        
        image_x= np.squeeze(traindata_batch_x[batch_i,:,:,:]).astype('uint8')
        image_y= np.squeeze(traindata_batch_y[batch_i,:,:,:]).astype('uint8')
        
        #print(image_x)
        
        traindata_batch_x[batch_i,:,:,:]=np.array(randomColor(Image.fromarray(image_x)))
        traindata_batch_y[batch_i,:,:,:]=np.array(randomColor(Image.fromarray(image_y)))
        
        
        '''
        rand_num=np.random.randint(0,2)
        
        if rand_num :
          traindata_batch_x[batch_i,:,:,:]=randomGaussian(np.squeeze(traindata_batch_x[batch_i,:,:,:]), mean=0.2, sigma=0.3)
          traindata_batch_y[batch_i,:,:,:]=randomGaussian(np.squeeze(traindata_batch_y[batch_i,:,:,:]), mean=0.2, sigma=0.3)
        '''  
        
        
      traindata_batch_x=np.float32((1/255)*traindata_batch_x.astype('float32'))
      traindata_batch_y=np.float32((1/255)*traindata_batch_y.astype('float32'))
    
      traindata_batch_x=np.minimum(np.maximum(traindata_batch_x,0),1)
      traindata_batch_y=np.minimum(np.maximum(traindata_batch_y,0),1)  
        
        
      return traindata_batch_x,traindata_batch_y


def load_LF_valdata(dir_LFimages):   
       traindata_all_3=np.zeros((len(dir_LFimages), 9, 9, 512, 512, 3),np.uint16)
       traindata_all=np.zeros((len(dir_LFimages), 9, 9, 512, 512,3),np.uint16)
       traindata_y=np.zeros((len(dir_LFimages),512*9,512,3),np.uint16)
       traindata_x=np.zeros((len(dir_LFimages),512,512*9,3),np.uint16)
        
       traindata_label=np.zeros((len(dir_LFimages), 512, 512),np.float32)
       
       image_id=0
       
       for dir_LFimage in dir_LFimages:
          #print(dir_LFimage)
        
         traindata_all_3[image_id,:,:,:,:,:]=read_lightfield(dir_LFimage)
         traindata_all[image_id,:,:,:,:,:]=traindata_all_3[image_id,:,:,:,:,:]
        
         temp1=traindata_all[image_id,:,4,:,:,:].copy()
         temp1=temp1.transpose(2,1,0,3)
         temp1=temp1.transpose(1,0,2,3)
         traindata_x[image_id,:,:,:]=temp1.reshape(512,512*9,3)
        
         temp2=traindata_all[image_id,4,:,:,:,:].copy()
         temp2=temp2.transpose(1,0,2,3)
         traindata_y[image_id,:,:,:]=temp2.reshape(512*9,512,3)
         
         
         
         
         traindata_label[image_id,:,:]=np.float32(read_pfm(dir_LFimage+'/gt_disp_lowres.pfm'))  
         
         image_id=image_id+1
       #traindata_label = traindata_label[:,:,:,np.newaxis]
        
       return traindata_x, traindata_y, traindata_label,len(dir_LFimages)    #,traindata_x_new,traindata_y_new



