#! /usr/bin/python
import tensorflow as tf
import numpy as np
from model_base import *
import mnist

class mnist_nn(model_base):
    
    def __init__(self,reader):
        self.base = super(mnist_nn,self)
        self.base.__init__(reader)

    def decl_model(self):
        x = self.decl_placeholder("x",[None,784])
        y_ = self.decl_placeholder("y_",[None,10])
        w = self.decl_weight([784,100])
        b = self.decl_bias([100])
        m = tf.matmul(x,w)
        b = m + b + tf.cos(m)
        h = tf.nn.relu(m)

        # h =  self.decl_full_conn_layer("hidden0",x,[784,100],[100])
        loss = self.decl_full_conn_softmax_crossentry_layer("hidden1",h,[100,10],[10],y_)
        return 1000,loss,x,y_,tf.train.GradientDescentOptimizer(self.learn_rate)


if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/train-images.idx3-ubyte","/home/lr/workspace/python/ai/data/train-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_nn_cos/model.ckpt")
    model = mnist_nn(m)
    model.plot(model.train(10),"count","loss")