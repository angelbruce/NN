#! /usr/bin/python

import tensorflow as tf
import numpy as np
from check_base import *
import mnist

class mnist_cnn_test(check_base):
    
    def __init(self,reader):
        self.base = super(mnist_cnn_test,self)
        self.base.__init__(reader)

    def decl_predict(self):
        batch = 1
        x = self.decl_placeholder("x",[None,784])
        y_ = self.decl_placeholder("y_",[None,10])
        x1 = tf.reshape(x,[-1,28,28,1])
        c1 = self.decl_conv2d_layer("c1",x1,[5,5,1,32],[32])
        p1 = self.decl_max_pool("max_pool1",c1)
        c2 = self.decl_conv2d_layer("c2",p1,[5,5,32,64],[64])
        p2 = self.decl_max_pool("max_pool2",c2)
        flat = tf.reshape(p2,[-1,7 * 7 * 64])
        dense = self.decl_full_conn_layer("dense",flat,[7 * 7 * 64,1024],[1024])
        y = self.decl_full_conn_softmax_layer("fc2",dense,[1024,10],[10])
        return 1,y,x,y_

if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/t10k-images.idx3-ubyte","/home/lr/workspace/python/ai/data/t10k-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_cnn/model.ckpt")
    model = mnist_cnn_test(m)
    model.check()
