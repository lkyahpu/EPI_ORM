# -*- coding: utf-8 -*-
"""


@author: lky
"""
import numpy as np
import imageio
from file_io2 import write_pfm , read_pfm
def display_current_output(val_output, valdata_label, iter00, directory_save):
    
    
    #train_output=np.argmax(train_output, axis=1).astype(np.uint8) 
    #traindata_label=np.argmax(traindata_label, axis=1).astype(np.uint8) 


    
    val_out_disp= np.zeros((482,1*482),dtype=np.float32)
    val_out_disp= val_output.reshape(482,482*1).astype(np.float32)
    val_out_label_disp=np.zeros((482,1*482),dtype=np.float32)    
    val_out_label_disp=valdata_label.reshape(482,482*1).astype(np.float32)
    

    val_diff=np.abs(val_out_disp-val_out_label_disp)  
    val_bp  = (  val_diff >= 0.07  )
    

    #val_out=np.zeros((482,1*482),dtype=np.float32)
    #val_out_label=np.zeros((482,1*482),dtype=np.float32)

    #val_out[:,0:482]=((val_out_disp[:,0:482]+2.2)/3.6).astype(np.float32)  
    #val_out[:,482:2*482]=((val_out_disp[:,482:2*482]+1.6)/3.1).astype(np.float32)
    #val_out[:,2*482:3*482]=((val_out_disp[:,2*482:3*482]+1.9)/3.8).astype(np.float32)
    #val_out[:,482:2*482]=((val_out_disp[:,482:2*482]+2.0)/3.7).astype(np.float32)


    

    #val_out_label[:,0:482]=((val_out_label_disp[:,0:482]+2.2)/3.6).astype(np.float32)  
    #val_out_label[:,482:2*482]=((val_out_label_disp[:,482:2*482]+1.6)/3.1).astype(np.float32)
    #val_out_label[:,2*482:3*482]=((val_out_label_disp[:,2*482:3*482]+1.9)/3.8).astype(np.float32)
    #val_out_label[:,482:2*482]=((val_out_label_disp[:,482:2*482]+2.0)/3.7).astype(np.float32)


    #val_out=np.squeeze(train_output).reshape(160*3,160*3)

    #val_label=traindata_label.reshape(512,512)
    
    
    
    
    #val_out_label= traindata_label.reshape(162,162,3,3).transpose(0,2,1,3).reshape(162*3,162*3)[2:-2,2:-2]
    #val_out_label= traindata_label.reshape(482,482,1,1).transpose(0,2,1,3).reshape(482*1,482*1)
    #val_out_label= traindata_label.reshape(160*3,160*3)
    #sz=len(traindata_label)
    #train_output=np.squeeze(train_output)
    #if(len(traindata_label.shape)>3 and traindata_label.shape[-1]==9): # traindata
     #   pad1_half=int(0.5*(np.size(traindata_label,1)-np.size(train_output,1)))
      #  train_label482=traindata_label[:,15:-15,15:-15,4,4]
   # else: # valdata
    #pad1_half=int(0.5*(np.size(traindata_label,1)-np.size(train_output,1)))
   

    
    '''      
    val_output482_all=np.zeros((2*482,482),np.float32)        
    val_output482_all[0:482,:]=(val_out_label+1.9)/3.8
    val_output482_all[482:2*482,:]=(val_out+1.9)/3.8
    '''

    val_output482_all=np.zeros((2*482,1*482),np.float32)        
    val_output482_all[0:482,:]= val_out_label_disp
    val_output482_all[482:2*482,:]= val_out_disp

    #imageio.imsave('val_output482_all.png',val_output482_all)
    #exit(0)
    val_output482_all=((val_output482_all+1.6)/3.1).astype(np.float32)  
    
    val_output482_all=(np.minimum(np.maximum(val_output482_all,0),1)*255).astype(np.uint8)

    #val_output482_all=val_output482_all.astype(np.uint8)
    
    #write_pfm(train_output482_all,directory_save+'/train_iter%05d.pfm' % (iter00))       
    
    #imageio.imsave(directory_save+'/train_iter%05d.png' % (iter00), np.squeeze(train_output482_all))
    
    return val_diff, val_bp, val_output482_all
