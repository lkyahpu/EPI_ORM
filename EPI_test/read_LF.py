#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:53:23 2019

@author: lky
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:51:55 2019

@author: lky
"""
import numpy as np
from imageio import imread 
import os
def read_all_LF(num_x,num_y,fpath):
    
    
    height=np.shape(imread(fpath))[0]
    width =np.shape(imread(fpath))[1]
    channel=np.shape(imread(fpath))[2]
    
    valdata_x=np.zeros((height//num_x,(width//num_y)*9,3),np.uint8)
    valdata_y=np.zeros(((height//num_x)*9,width//num_y,3),np.uint8)
    
    
    light_field = np.zeros((9,9,height//num_x,width//num_y,3),dtype=np.uint8)
    temp=imread(fpath).reshape(height//num_x,num_x,width//num_y,num_y,channel)
    temp=temp.transpose(1,0,3,2,4)
    light_field = temp.transpose(0,2,1,3,4)[0:9,0:9,:,:,0:3]
    
    
    temp1=light_field[4,:,:,:,:].copy().transpose(1,0,2,3)
    valdata_y[:,:,:]=temp1.reshape((height//num_x)*9,width//num_y,3)
    
    temp2=light_field[:,4,:,:,:].copy().transpose(2,1,0,3)
    temp2=temp2.transpose(1,0,2,3)
    valdata_x[:,:,:]=temp2.reshape(height//num_x,(width//num_y)*9,3)
    
         
    
    input_size1=9
    input_size2=29
    


    numx=width//num_y-28
    numy=height//num_x-28
    num=0
    
   

    valdata_batch_x=np.zeros((1*numx*numy,input_size1,input_size2,3),dtype=np.float32)
    valdata_batch_y=np.zeros((1*numx*numy,input_size1,input_size2,3),dtype=np.float32)
    valdata_batch_y_tmp=np.zeros((1*numx*numy,input_size2,input_size1,3),dtype=np.float32) 
    

    for kk in range(0,numy,1):
        for jj in range(0,numx,1):     
          valdata_batch_x[num,:,:,:]=valdata_y[ (kk+14)*9+4-(input_size1//2) : (kk+14)*9+4+(input_size1//2)+1, (jj+14)-(input_size2//2):(jj+14)+(input_size2//2)+1, :]
          valdata_batch_y_tmp[num,:,:,:]=valdata_x[ (kk+14)-(input_size2//2) : (kk+14)+(input_size2//2)+1, (jj+14)*9+4-(input_size1//2):(jj+14)*9+4+(input_size1//2)+1, :]
          valdata_batch_y[num,:,:,:]= np.copy(np.rot90(valdata_batch_y_tmp[num,:,:,:],1,(0,1)))
         
          num=num+1
          
    valdata_batch_x=np.float32((1/255)*valdata_batch_x)
    valdata_batch_y=np.float32((1/255)*valdata_batch_y)

    

    valdata_batch_x=np.minimum(np.maximum(valdata_batch_x,0),1)
    valdata_batch_y=np.minimum(np.maximum(valdata_batch_y,0),1)    

    
    '''
    np.save('/media/lky/软件/LKY/EPI_test/valdata_batch_x.npy',valdata_batch_x)
    np.save('/media/lky/软件/LKY/EPI_test/valdata_batch_y.npy',valdata_batch_y)
    np.save('/media/lky/软件/LKY/EPI_test/valdata_batch_xd.npy',valdata_batch_xd)
    np.save('/media/lky/软件/LKY/EPI_test/valdata_batch_yd.npy',valdata_batch_yd)
    
    print(numy,numx)
    '''
    return valdata_batch_x,valdata_batch_y,numy,numx    #   ,valdata_batch_xd,valdata_batch_yd
    
    
def read_pinhole_LF(num_x,num_y,fpath):
    #light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.uint8)
    #light_field = np.zeros((9,9,height//num_x,width//num_y,3),dtype=np.uint8)
    #LF_path=os.listdir(fpath)
    #temp=np.shape(imread(os.path.join(fpath,LF_path[0])))
    files = [f.name for f in os.scandir(fpath)]
    imgs = [f for f in files if (f.endswith('.png') or f.endswith(
            '.jpg') or f.endswith('.jpeg')) and 'normals' not in f and
            'mask' not in f and 'objectids' not in f and 'unused' not in f]
    imgs.sort()
    temp=np.shape(imread(os.path.join(fpath,imgs[0])))
    height=temp[0]
    width=temp[1]
    light_field = np.zeros((num_x,num_y,height,width,3),dtype=np.uint8)
    for idx, view in enumerate(imgs):
       light_field[idx//num_x,idx%num_y,:,:,:]=imread(os.path.join(fpath, view))[:,:,0:3]
        
    LF = np.zeros((9,9,height,width,3),dtype=np.uint8)
    LF = light_field[0:9,0:9,:,:,:] 
    
    valdata_y=np.zeros((height*9,width,3),np.uint8)
    valdata_x=np.zeros((height,width*9,3),np.uint8)
    
    
    
    temp1=LF[4,:,:,:,:].copy().transpose(1,0,2,3)
    valdata_y[:,:,:]=temp1.reshape(height*9,width,3)
    
    temp2=LF[:,4,:,:,:].copy().transpose(2,1,0,3)
    temp2=temp2.transpose(1,0,2,3)
    valdata_x[:,:,:]=temp2.reshape(height,width*9,3)
    
    input_size1=9
    input_size2=29

    numx=width-28
    numy=height-28
    num=0

    valdata_batch_x=np.zeros((1*numx*numy,input_size1,input_size2,3),dtype=np.float32)
    valdata_batch_y=np.zeros((1*numx*numy,input_size1,input_size2,3),dtype=np.float32)
    valdata_batch_y_tmp=np.zeros((1*numx*numy,input_size2,input_size1,3),dtype=np.float32) 

    for kk in range(0,numy,1):
        for jj in range(0,numx,1):     
          valdata_batch_x[num,:,:,:]=valdata_y[ (kk+14)*9+4-(input_size1//2) : (kk+14)*9+4+(input_size1//2)+1, (jj+14)-(input_size2//2):(jj+14)+(input_size2//2)+1, :]
          valdata_batch_y_tmp[num,:,:,:]=valdata_x[ (kk+14)-(input_size2//2) : (kk+14)+(input_size2//2)+1, (jj+14)*9+4-(input_size1//2):(jj+14)*9+4+(input_size1//2)+1, :]
          valdata_batch_y[num,:,:,:]= np.copy(np.rot90(valdata_batch_y_tmp[num,:,:,:],1,(0,1)))
         
          num=num+1
          
    valdata_batch_x=np.float32((1/255)*valdata_batch_x)
    valdata_batch_y=np.float32((1/255)*valdata_batch_y)
  

    valdata_batch_x=np.minimum(np.maximum(valdata_batch_x,0),1)
    valdata_batch_y=np.minimum(np.maximum(valdata_batch_y,0),1)    

    
    '''
    np.save('/media/lky/软件/LKY/EPI_test/valdata_batch_x.npy',valdata_batch_x)
    np.save('/media/lky/软件/LKY/EPI_test/valdata_batch_y.npy',valdata_batch_y)
    np.save('/media/lky/软件/LKY/EPI_test/valdata_batch_xd.npy',valdata_batch_xd)
    np.save('/media/lky/软件/LKY/EPI_test/valdata_batch_yd.npy',valdata_batch_yd)
    
    print(numy,numx)
    '''
    return valdata_batch_x,valdata_batch_y,numy,numx    #   ,valdata_batch_xd,valdata_batch_yd
    






    
    
    
    
    
    
    
    
    
    