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
        x = self.decl_placeholder("x",[None,3072])
        y_ = self.decl_placeholder("y_",[None,10])
        x1 = tf.reshape(x,[-1,32,32,3])
        c1 = self.decl_conv2d_layer("c1",x1,[5,5,3,16],[16],padding='VALID')
        p1 = self.decl_max_pool("max_pool1",c1)
        c2 = self.decl_conv2d_layer("c2",p1,[5,5,16,64],[64],padding='VALID')
        p2 = self.decl_max_pool("max_pool2",c2)
        c3 = self.decl_conv2d_layer("c3",p2,[5,5,64,256],[256],padding='VALID')
        print(c1,p1,c2,p2,c3)
        flat = tf.reshape(c3,[-1,1 * 1 * 256])
        y = self.decl_full_conn_layer("y",flat,[256,10],[10],isActive=False)
        return 1,y,x,y_

if __name__ == "__main__":
    print("#"*30)
    m = cifar.cifar("/home/lr/workspace/python/ai/data/cifar-10-batches-py/","/home/lr/workspace/python/ai/model/cifar_cnn/model.ckpt",is_test=True)
    model = cifar_cnn_test(m)
    model.check()
