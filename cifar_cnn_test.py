#! /usr/bin/python

import tensorflow as tf
import numpy as np
from cifar_check_base import *
import cifar

class cifar_cnn_test(cifar_check_base):
    
    def __init(self,reader):
        self.base = super(cifar_cnn_test,self)
        self.base.__init__(reader)

    def decl_predict(self):
        batch = 1
        x = self.decl_placeholder("x",[None,3072])
        y_ = self.decl_placeholder("y_",[None,10])
        
        x1 = tf.reshape(x,[batch,32,32,3])

        c1 = self.decl_conv2d_layer("c1",x1,[5,5,3,32],[32])
        p1 = self.decl_max_pool("max_pool1",c1,padding='SAME')

        c2 = self.decl_conv2d_layer("c2",p1,[5,5,32,64],[64])
        p2 = self.decl_max_pool("max_pool2",c2,padding='SAME')

        c3 = self.decl_conv2d_layer("c3",p2,[5,5,64,64],[64])
        p3 = self.decl_max_pool("max_pool3",c3,padding='SAME')
      
        flat = tf.reshape(p3,[-1,4 * 4 * 64])
        dense = self.decl_full_conn_layer("dense",flat,[4 * 4 * 64,256],[256])
        y = self.decl_full_conn_softmax_layer("fc2",dense,[256,10],[10])
        return 1,y,x,y_

if __name__ == "__main__":
    print("#"*30)
    m = cifar.cifar("/home/lr/workspace/python/ai/data/cifar-10-batches-py/","/home/lr/workspace/python/ai/model/cifar_cnn/model.ckpt",is_test=True)
    model = cifar_cnn_test(m)
    model.check()
