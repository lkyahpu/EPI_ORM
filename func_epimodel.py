# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:54:06 2018

@author: lky
"""
from keras.utils import plot_model


from keras import initializers
from keras.optimizers import RMSprop ,Adagrad
from keras.layers.merge import concatenate 
from keras.layers.core import  Activation,Reshape ,Permute
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, ZeroPadding2D ,Cropping2D
from keras.layers.normalization  import BatchNormalization
from keras.layers import Input , Add ,add,average, GlobalAvgPool2D,Dense  ,Lambda  ,dot
from keras.models import Model
import tensorflow as tf
from keras.initializers import glorot_uniform ,TruncatedNormal
from keras.models import load_model
from keras import backend as K
from keras import regularizers


def layer2_1(input_dim1,input_dim2,input_dim3,filt_num):
    
     seq = Sequential()   
       
     seq.add(Conv2D(int(filt_num),(2, 2),strides=(1, 1), input_shape=(input_dim1, input_dim2, input_dim3), data_format='channels_last' ,padding='valid', name='S2_1_c1%d_new'% (0)  ))
     seq.add(Activation('relu', name='S2_1_relu1%d'% (0) ))       
     seq.add(Conv2D(int(filt_num),(2, 2),strides=(1, 1),padding='valid' ,name='S2_1_c2_%d'% (0) ))    
     seq.add(BatchNormalization(axis=-1, name='S2_1_BN%d'% (0) ))
     seq.add(Activation('relu', name='S2_1_relu2_%d'% (0))) 
     
     for i in range(1,4):   
      seq.add(Conv2D(int(filt_num),(2, 2),strides=(1, 1),  data_format='channels_last' ,padding='valid', name='S2_1_c1%d'% (i)  ))
      seq.add(Activation('relu', name='S2_1_relu1%d'% (i) ))       
      seq.add(Conv2D(int(filt_num),(2, 2),strides=(1, 1),padding='valid' ,name='S2_1_c2_%d'% (i) ))    
      seq.add(BatchNormalization(axis=-1, name='S2_1_BN%d'% (i) ))
      seq.add(Activation('relu', name='S2_1_relu2_%d'% (i)))         
       
     return seq    

    
def layer2_2(input_dim1,input_dim2,input_dim3,filt_num):
    
     seq = Sequential()   
    
     seq.add(Conv2D(int(filt_num),(2, 2),strides=(1, 1), input_shape=(input_dim1, input_dim2, input_dim3), data_format='channels_last' ,padding='valid', name='S2_2_c1%d_new'% (0)  )) 
     seq.add(Activation('relu', name='S2_2_relu1%d'% (0) ))     
     seq.add(Conv2D(int(filt_num),(2, 2),strides=(1, 1),padding='valid' ,name='S2_2_c2_%d'% (0))) 
     seq.add(BatchNormalization(axis=-1, name='S2_2_BN%d'% (0)))
     seq.add(Activation('relu', name='S2_2_relu2_%d'% (0))) 
     for i in range(1,4):    
      seq.add(Conv2D(int(filt_num),(2, 2),strides=(1, 1),  data_format='channels_last' ,padding='valid', name='S2_2_c1%d'% (i)  )) 
      seq.add(Activation('relu', name='S2_2_relu1%d'% (i) ))     
      seq.add(Conv2D(int(filt_num),(2, 2),strides=(1, 1),padding='valid' ,name='S2_2_c2_%d'% (i))) 
      seq.add(BatchNormalization(axis=-1, name='S2_2_BN%d'% (i)))
      seq.add(Activation('relu', name='S2_2_relu2_%d'% (i))) 
       
     return seq  
    
  
def layer_mid(name_layer,filt_num):
    seq = Sequential()
    for i in range(1,4):       
      seq.add(Conv2D(int(filt_num),(1,2), strides=(1, 1),padding='valid', name=name_layer+'cov_mid_c%d'%(i*2-1))) 
      seq.add(Activation('relu', name=name_layer+'relu_mid_%d'%(i*2-1)))
      seq.add(Conv2D(int(filt_num),(1,2), strides=(1, 1),padding='valid' , name=name_layer+'cov_mid_c%d'%(i*2))) 
      seq.add(BatchNormalization(axis=-1, name=name_layer+'BN_mid_%d'%(i)))
      seq.add(Activation('relu', name=name_layer+'relu_mid_%d'%(i*2))) 

    return seq

  
def ORM(input_x,filt_num,h,w,name_layer,flag):
       
    mid_merged_u1 = Conv2D(int(filt_num),(1,1), strides=(1, 1),padding='valid', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None) ,name='ORM_1x1_1_'+name_layer)(input_x)
    mid_merged_v1 = Conv2D(int(filt_num),(1,1), strides=(1, 1),padding='valid', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None) ,name='ORM_1x1_2_'+name_layer)(input_x)
    
    mid_merged_u1 = Reshape((h*w,128))(mid_merged_u1)
    mid_merged_v1 = Reshape((h*w,128))(mid_merged_v1)
    mid_merged_v1 = Permute((2,1))(mid_merged_v1)
    mid_merged_uv1 = Lambda(lambda x: K.batch_dot(x[0],x[1],axes=[2,1]))([mid_merged_u1, mid_merged_v1]) 
    
    if flag:
     mid_merged_uv1 = Lambda(lambda x: K.expand_dims(x,3))(mid_merged_uv1) 
     mid_merged_uv1 = Cropping2D(cropping=(((h*w-29)//2, (h*w-29)//2), (0, 0)))(mid_merged_uv1)
     mid_merged_uv1 = Lambda(lambda x: K.squeeze(x,3))(mid_merged_uv1) 
     mid_merged_uv1 = Permute((2,1))(mid_merged_uv1)
    
    mid_merged_uv1 = Reshape((h,w,w))(mid_merged_uv1)
    mid_merged_uv1 = Activation('relu')(mid_merged_uv1)
    mid_merged_uv1 = concatenate([input_x,mid_merged_uv1])     
    
    return mid_merged_uv1
  
  

def res_block_1(X,filt_num):
 
   
    X = Conv2D(int(filt_num), kernel_size = (1,1), strides = (1,1), padding = 'valid',name='res1_1x1_new')(X)
    X = Activation('relu')(X)
    
    for i in range(1,7):
     X_shortcut = X
     X = Conv2D(int(filt_num), kernel_size = (1,2), strides = (1,1), padding = 'valid',name='res1_c%d'%(i*2-1))(X)
     X = Activation('relu')(X)
     X = Conv2D(int(filt_num), kernel_size = (1,2), strides = (1,1), padding = 'valid',name='res1_c%d'%(i*2))(X)
     X = BatchNormalization(axis = 3)(X)
     X = Activation('relu')(X)
     X_shortcut=Cropping2D(cropping=((0, 0), (1, 1)))(X_shortcut)
     X = add([X, X_shortcut])
    
    return X
    

def res_block_2(X,filt_num):
 
    
    X = Conv2D(int(filt_num), kernel_size = (1,1), strides = (1,1), padding = 'valid',name='res2_1x1_new')(X)
    X = Activation('relu')(X)
    
    for i in range(1,7):
     X_shortcut = X
     X = Conv2D(int(filt_num), kernel_size = (1,2), strides = (1,1), padding = 'valid',name='res2_c%d'%(i*2-1))(X)
     X = Activation('relu')(X)
     X = Conv2D(int(filt_num), kernel_size = (1,2), strides = (1,1), padding = 'valid',name='res2_c%d'%(i*2))(X)
     X = BatchNormalization(axis = 3)(X)
     X = Activation('relu')(X)
     X_shortcut=Cropping2D(cropping=((0, 0), (1, 1)))(X_shortcut)
     X = add([X, X_shortcut])
    
    return X


def define_epi(sz_input,sz_input2,filt_num,learning_rate):

    input_x = Input(shape=(sz_input,sz_input2,3), name='input_x')
    input_y = Input(shape=(sz_input,sz_input2,3), name='input_y')
    
    mid_merged_uv1=ORM(input_x,filt_num,9,29,'input_x',True)
    mid_merged_uv2=ORM(input_y,filt_num,9,29,'input_y',True)
    
    mid_input_x=layer2_1(sz_input,sz_input2,32,int(filt_num))(mid_merged_uv1)
    mid_input_y=layer2_2(sz_input,sz_input2,32,int(filt_num))(mid_merged_uv2)    
    
    mid_input_x=layer_mid('input_x',filt_num)(mid_input_x)
    mid_input_y=layer_mid('input_y',filt_num)(mid_input_y)
    
    mid_uv1=ORM(mid_input_x,filt_num,1,15,'input_x_mid',False)
    mid_uv2=ORM(mid_input_y,filt_num,1,15,'input_y_mid',False)
     
    '''
    tmp1 = Conv2D(int(filt_num),(1,2), strides=(1, 1),padding='valid',data_format='channels_last', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None) ,name='S3_c_1_last'  )(mid_merged_uv1)# pow(25/23,2)*12*(maybe7?) 43 3
    tmp1 = Activation('relu', name='S_3_relu1_last' )(tmp1)
    tmp1 = Conv2D(int(filt_num),(1,2), strides=(1, 1),padding='valid',kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None) , name='S3_c2_last')(tmp1)
    tmp1 = BatchNormalization(axis=-1, name='S_3_BN_last')(tmp1)
    tmp1 = Activation('relu', name='S3_relu2_last')(tmp1)
    
    tmp1=Conv2D(int(filt_num),(1,2), strides=(1, 1),padding='valid', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None) ,name='S1_last_c1')(tmp1)     
    tmp1=Activation('relu', name='S1_last_relu')(tmp1)
    output1=Conv2D(1,(1,2),strides=(1, 1), padding='valid', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None) ,name='S1_last_c2') (tmp1)
    '''
    '''
    tmp2 = Conv2D(int(filt_num),(1,2), strides=(1, 1),padding='valid',data_format='channels_last', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None) ,name='S3_c1_last_new'  )(mid_merged_uv2)# pow(25/23,2)*12*(maybe7?) 43 3
    tmp2 = Activation('relu', name='S3_relu1_last_new' )(tmp2)
    tmp2 = Conv2D(int(filt_num),(1,2), strides=(1, 1),padding='valid',kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None) , name='S3_c2_last_')(tmp2)
    tmp2 = BatchNormalization(axis=-1, name='S_3_BN_last_')(tmp2)
    tmp2 = Activation('relu', name='S3_relu2_last_')(tmp2)
   
    
    tmp2=Conv2D(int(filt_num),(1,2), strides=(1, 1),padding='valid', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None) ,name='S2_last_c1')(tmp2)     
    tmp2=Activation('relu', name='S2_last_relu')(tmp2)
    output2=Conv2D(1,(1,2),strides=(1, 1), padding='valid', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None) ,name='S2_last_c2') (tmp2)
    '''
    
    mid_input_x_orm=res_block_1(mid_uv1,filt_num)
    mid_input_y_orm=res_block_2(mid_uv2,filt_num)
   
    
    mid_merged=concatenate([mid_input_x_orm, mid_input_y_orm],  name='mid_merged')  
    
    
    mid_merged=Conv2D(int(filt_num),(1,2), strides=(1, 1),padding='valid', name='s3_last_c5')(mid_merged)
    mid_merged=Activation('relu', name='S3_relu1_last_5')(mid_merged)
    mid_merged=Conv2D(int(filt_num),(1,2), strides=(1, 1),padding='valid', name='s3_last_c6')(mid_merged)
    mid_merged=BatchNormalization(axis=-1, name='S3_BN_last_3')(mid_merged)
    mid_merged=Activation('relu', name='S3_relu1_last_6')(mid_merged)
    
    mid_merged=Conv2D(2*int(filt_num),(1,1), strides=(1, 1),padding='valid',name='S3_c2%d'%(1) )(mid_merged)
    mid_merged=Activation('relu', name='S3_relu_%d' %(1))(mid_merged)
    output=Conv2D(1,(1,1), strides=(1, 1),  padding='valid' ,  name='loss_output')(mid_merged)
    
    
    model_512 = Model(inputs = [input_x, input_y ], outputs = output)   #output1,output2,
    opt = RMSprop(lr=learning_rate)
    #RMSprop(lr=learning_rate)
    model_512.compile(optimizer=opt, loss= 'mae')  #'mae','mae', loss_weights=[0.5, 0.5, 1.]
    
    #model_512.summary() 
    
    return model_512
  
  

filt_num=128
learning_rate=0.1**4

sz_input=9
sz_input2=29

model=define_epi(sz_input,sz_input2,filt_num,learning_rate)

                        
#plot_model(model, to_file='model_test.png',show_shapes=True)

print(model.summary())