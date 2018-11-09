#!/usr/bin/python

import tensorflow as tf
import numpy as np
from model_base import *
import cifar

class cifar_cnn(model_base):

    def __init__(self,reader):
        self.base =  super(cifar_cnn,self)
        self.base.__init__(reader)

    def decl_model(self):
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
        loss = self.decl_softmax_corssentry_loss(y,y_)
        return 100,loss,x,y_,y,tf.train.RMSPropOptimizer(self.learn_rate,0.5)

        

if __name__ == "__main__":
    print("#"*30)
    m = cifar.cifar("/home/lr/workspace/python/ai/data/cifar-10-batches-py/","/home/lr/workspace/python/ai/model/cifar_cnn/model.ckpt")
    model = cifar_cnn(m)
    model.plot(model.train(100,0.001),"count","loss","accuracy")
    