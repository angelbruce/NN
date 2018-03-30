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
        batch = 100
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
        dropout = self.decl_drop_out_layer(dense,0.5,"dropout")
        loss = self.decl_full_conn_softmax_crossentry_layer("fc2",dropout,[256,10],[10],y_)
        return batch,loss,x,y_,tf.train.AdamOptimizer(1e-4)

        

if __name__ == "__main__":
    print("#"*30)
    m = cifar.cifar("/home/lr/workspace/python/ai/data/cifar-10-batches-py/","/home/lr/workspace/python/ai/model/cifar_cnn/model.ckpt")
    model = cifar_cnn(m)
    model.plot(model.train(100,0.001),"count","loss")