#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:13:15 2019

@author: lky
"""
import time
import matplotlib.pyplot as plt
import pickle
if __name__ == '__main__':	
    n=100
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
    Epoch=[x for x in range(0,n)]
    train_loss=load_variavle('train_loss.pkl')
    test_loss=load_variavle('test_loss.pkl')
    
    #是否保存


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