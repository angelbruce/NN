#! /usr/bin/python

import tensorflow as tf
import numpy as np
from check_base import *
import mnist

class mnist_nn_test(check_base):
    
    def __init(self,reader):
        self.base = super(mnist_nn_test,self)
        self.base.__init__(reader)

    def decl_predict(self):
        x = self.decl_placeholder("x",[None,784])
        y_ = self.decl_placeholder("y_",[None,10])
        w = self.decl_weight([784,100])
        b = self.decl_bias([100])
        m = tf.matmul(x,w)
        b = m + b + tf.cos(m)
        h0 = tf.nn.relu(m)
        h1 =  self.decl_full_conn_layer("hidden1",h0,[100,10],[10])
        p = tf.nn.softmax(h1)
        return 1,p,x, y_


if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/t10k-images.idx3-ubyte","/home/lr/workspace/python/ai/data/t10k-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_nn_cos/model.ckpt")
    model = mnist_nn_test(m)
    model.check()
