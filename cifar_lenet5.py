#!/usr/bin/python

import tensorflow as tf
import numpy as np
from model_base import *
import cifar

class cifar_lenet5(model_base):

    def __init__(self,reader):
        self.base =  super(cifar_lenet5,self)
        self.base.__init__(reader)

    def decl_model(self):
        batch = 100
        x = self.decl_placeholder("x",[None,3072])
        y_ = self.decl_placeholder("y_",[None,10])
        x1 = tf.reshape(x,[batch,32,32,3])

        print(x1);
        c1 = self.decl_conv2d_layer("c1",x1,[5,5,3,6],[6])
        print(c1)
        p1 = self.decl_max_pool("max_pool1",c1,padding='SAME')
        print(p1)
        c2 = self.decl_conv2d_layer("c2",p1,[5,5,6,16],[16])
        print(c2)
        p2 = self.decl_max_pool("max_pool2",c2,padding='SAME')
        print(p2)
        
        c3 = self.decl_conv2d_layer("c3",p2,[5,5,16,16],[16])
        print(c3)
        p3 = self.decl_max_pool("max_pool3",c3,padding='SAME')
        print(p3)
      
        flat = tf.reshape(p3,[-1,4 * 4 * 16])
        dense = self.decl_full_conn_layer("dense",flat,[4 * 4 * 16,84],[84])
        loss = self.decl_full_conn_softmax_crossentry_layer("fc2",dense,[84,10],[10],y_)
        return batch,loss,x,y_,None

        

if __name__ == "__main__":
    print("#"*30)
    m = cifar.cifar("/home/lr/workspace/python/ai/data/cifar-10-batches-py/","/home/lr/workspace/python/ai/model/cifar_lenet5/model.ckpt")
    model = cifar_lenet5(m)
    model.plot(model.train(1,0.01),"count","loss")
