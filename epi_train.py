# -*- coding: utf-8 -*-


from __future__ import print_function

#from tensorflow import keras
#from keras.models import load_model
import keras
from func_epimodel import define_epi
from func_savedata import display_current_output
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#import h5py
import os
import time
import imageio
import datetime
import threading
import pickle

from load_LF import load_LFdata

from load_LF import generate_traindata
from load_LF import generate_valdata




if __name__ == '__main__':	
    
    ''' 
    We use fit_generator to train EPINET, 
    so here we defined a generator function.
    '''
    n=5000
    #train_loss=np.zeros(n,np.float32)
    #test_loss=np.zeros(n,np.float32)
    Epoch=np.zeros(n,np.int16)
    
    
    
    train_loss=[]
    test_loss=[]
    
    def save_variable(v,filename):
      f=open(filename,'wb')
      pickle.dump(v,f)
      f.close()
      return filename
 
    def load_variavle(filename):
      f=open(filename,'rb')
      r=pickle.load(f)
      f.close()
      return r

    
    class threadsafe_iter:
        """Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.
        """
        def __init__(self, it):
            self.it = it
            self.lock = threading.Lock()
    
        def __iter__(self):
            return self
    
        def __next__(self):
            with self.lock:
                return self.it.__next__()
    
    
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
 
        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss_output_loss'))

    def threadsafe_generator(f):
        """A decorator that takes a generator function and makes it thread-safe.
        """
        def g(*a, **kw):
            return threadsafe_iter(f(*a, **kw))
        return g

    
    @threadsafe_generator
    def myGenerator(traindata_x, traindata_y, traindata_label,batch_size):    #,traindata_d_1,traindata_d_2
        while 1:
            #(traindata_x, traindata_y, traindata_label,_ )= load_LFdata(dir_LFimages)              

            (traindata_batch_x,traindata_batch_y,traindata_label_batchNxN) = generate_traindata(traindata_x, traindata_y, traindata_label,batch_size)  #,traindata_batch_d_1,traindata_batch_d_2  ,traindata_d_1,traindata_d_2    
            
            #(traindata_batch_x,traindata_batch_y) = data_augmentation_for_train(traindata_batch_x,traindata_batch_y, batch_size)
            
            traindata_label_batchNxN=traindata_label_batchNxN[:,:,:,np.newaxis] 
            

            yield([traindata_batch_x,traindata_batch_y],traindata_label_batchNxN)   #,traindata_batch_d_1,traindata_batch_d_2
    
            
    
    
    

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    
    If_trian_is = True;  
    
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"    
    
    #os.environ["CUDA_VISIBLE_DEVICES"]=""
    
   
    networkname='EPI_ORM_train'
    
    iter00=0; 
    
    load_weight_is=False;   

    model_learning_rate=0.1**4
    
    batch_size=128
    
    workers_num=2  # number of threads
    
    display_status_ratio=5000

       
    ''' 
    Define directory for saving checkpoint file & disparity output image
    '''       
    directory_ckp="epiorm_checkpoints/%s_ckp"% (networkname)     
    if not os.path.exists(directory_ckp):
        os.makedirs(directory_ckp)
        
    if not os.path.exists('epiorm_output/'):
        os.makedirs('epiorm_output/')   
    directory_t='epiorm_output/%s' % (networkname)    
    if not os.path.exists(directory_t):
        os.makedirs(directory_t)     
        
    txt_name='epiorm_checkpoints/lf_%s.txt' % (networkname)        
        
    ''' 
    Model for patch-wise training  
    '''
    #input_size1=9
    #input_size=(31,31,1)
   
    #model=define_epinet(input_size,  model_learning_rate)
    
    input_size1=9
    input_size2=29
    model_filt_num=128
    model_learning_rate=0.1**4
    
    #with tf.device('/cpu:0'):
    model=define_epi(input_size1,input_size2,
                            model_filt_num,
                            model_learning_rate)

    """ 
    load latest_checkpoint
    """
    if load_weight_is:
        list_name=os.listdir(directory_ckp)
        if(len(list_name)>=1):
            list1=os.listdir(directory_ckp)
            list_i=0
            for list1_tmp in list1:
                if(list1_tmp ==  'checkpoint'):
                    list1[list_i]=0
                    list_i=list_i+1   
                else:
                    list1[list_i]=int(list1_tmp.split('_')[0][4:])
                    list_i=list_i+1            
            list1=np.array(list1) 
            iter00=list1[np.argmax(list1)]+1
            ckp_name=list_name[np.argmax(list1)].split('.hdf5')[0]+'.hdf5'
            model.load_weights(directory_ckp+'/'+ckp_name)
            print("Network weights will be loaded from previous checkpoints \n(%s)" % ckp_name)
            
    #model.load_weights('iter0151_trainmse4.756_bp15.04.hdf5')
    
    
    """ 
    Write date & time 
    """
    f1 = open(txt_name, 'a')
    now = datetime.datetime.now()
    f1.write('\n'+str(now)+'\n\n')
    f1.close()    


    history = LossHistory()
    
    print('Load training data...')
    
    dir_LFimages  = ['/media/lky/文档/LKY/github/epinet-master/hci_dataset/additional/'+LFimage for LFimage in os.listdir('/media/lky/文档/LKY/github/epinet-master/hci_dataset/additional') if LFimage != 'license.txt']

    traindata_x, traindata_y, traindata_label = load_LFdata(dir_LFimages)

    my_generator = myGenerator(traindata_x,traindata_y, traindata_label,batch_size)   #,traindata_d_1,traindata_d_2
    
    print('Load test data...') 
    
    #dir_LFimages1=['training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']
    #dir_LFimages1=['E:/LKY/practice/EPI_ALL_REG_1/hci_dataset/'+LFimage for LFimage in dir_LFimages1]
    dir_LFimages1=['hci_dataset/training/cotton']

    valdata_fill_x, valdata_fill_y, valdata_label=generate_valdata(dir_LFimages1)
    #imageio.imsave('label.png', valdata_label)
    #exit(0)

    best_bad_pixel=100.0
    
    for iter02 in range(n):
        
        ''' Patch-wise training... start'''
        t0=time.time()
        
        batch_size=128
        
        ht=model.fit_generator(my_generator, steps_per_epoch = int(display_status_ratio), 
                            epochs = iter00+1, 
                            class_weight=None, 
                            initial_epoch=iter00, 
                            verbose=1,
                            workers=workers_num,
                            callbacks=[history])
        
        train_loss.append(ht.history['loss'])
        Epoch[iter02]=iter02
        iter00=iter00+1
    
        vald_output=model.predict([valdata_fill_x , valdata_fill_y ])   

        vald_error, vald_bp ,val_output482_all = display_current_output(vald_output, valdata_label, iter00, directory_t)

        print(np.average(np.abs(vald_error)))
        #print(loss)
        #print(accuracy)
        test_loss.append(np.average(np.abs(vald_error)))
        #test_loss.append(loss)
        
        training_mse_x100=100*np.average(np.square(vald_error))
        training_bad_pixel_ratio=100*np.average(vald_bp)
        
        print(training_bad_pixel_ratio)
        
        save_path_file_new=(directory_ckp+'/iter%04d_trainmse%.3f_bp%.2f.hdf5'  
                            % (iter00,training_mse_x100,
                              training_bad_pixel_ratio) )
               
        t1=time.time()    
        save_variable(train_loss,'train_loss.pkl')
        save_variable(test_loss,'test_loss.pkl')
        
        #save model weights if it get better results than previous one...
        if(training_bad_pixel_ratio < best_bad_pixel):
            best_bad_pixel = training_bad_pixel_ratio            
            #save_model =  model.sync_to_cpu()
            #save_model.save(save_path_file_new)

            model.save(save_path_file_new)
            imageio.imsave(directory_t+'/iter%04d_trainmse%.3f_bp%.3f.png' % (iter00,training_mse_x100,training_bad_pixel_ratio), val_output482_all)
            
            print("saved!!!")
        
    '''    
    plt.figure()  
    plt.plot(train_loss,label='train_loss')
    plt.plot(test_loss,label='val_loss')  
    plt.grid(True)
    plt.xlabel("Epoch")  
    plt.ylabel("train-val_loss")  
    plt.legend(loc="upper right")
    plt.title("train-val_loss")  
    plt.savefig("train_val_loss_%s.png"%(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))))     
    plt.show()        
    '''
