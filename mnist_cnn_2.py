#!/usr/bin/python

import tensorflow as tf
import numpy as np
from model_base import *
import mnist

class mnist_cnn_2(model_base):

    def __init__(self,reader):
        self.base =  super(mnist_cnn_2,self)
        self.base.__init__(reader)

    def decl_model(self):
        batch = 100
        x = self.decl_placeholder("x",[None,784])
        y_ = self.decl_placeholder("y_",[None,10])
        x1 = tf.reshape(x,[batch,28,28,1])
        c1 = self.decl_conv2d_layer("c1",x1,[5,5,1,32],[32])
        p1 = self.decl_max_pool("max_pool1",c1)
        c2 = self.decl_conv2d_layer("c2",p1,[5,5,32,64],[64])
        p2 = self.decl_max_pool("max_pool2",c2)
        flat = tf.reshape(p2,[-1,7 * 7 * 64])
        dense = self.decl_dense_layer("dense",flat,1024)
        loss = self.decl_full_conn_softmax_crossentry_layer("fc2",dense,[1024,10],[10],y_)
        return batch,loss,x,y_

if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/train-images.idx3-ubyte","/home/lr/workspace/python/ai/data/train-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_cnn_2/model.ckpt")
    model = mnist_cnn_2(m)
    model.plot(model.train(1,0.001),"count","loss")