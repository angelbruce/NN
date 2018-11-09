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
        h =  self.decl_full_conn_layer("hidden0",x,[784,100],[100])
        y = self.decl_full_conn_layer("hidden1",h,[100,10],[10],isActive=False)
        loss = self.decl_softmax_corssentry_loss(y,y_)
        return 1000,loss,x,y_,y,tf.train.GradientDescentOptimizer(self.learn_rate)


if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/train-images.idx3-ubyte","/home/lr/workspace/python/ai/data/train-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_nn/model.ckpt")
    model = mnist_nn(m)
    model.plot(model.train(1),"count","loss","accuracy")