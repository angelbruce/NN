#! /usr/bin/python
import tensorflow as tf
import numpy as np
from model_base import *
import cifar

class cifar_nn(model_base):
    
    def __init__(self,reader):
        self.base =  super(cifar_nn,self)
        self.base.__init__(reader)

    def decl_model(self):
        x = self.decl_placeholder("x",[None,3072])
        y_ = self.decl_placeholder("y_",[None,10])
        h0 =  self.decl_full_conn_layer("hidden0",x,[3072,100],[100])
        h1 =  self.decl_full_conn_layer("hidden1",h0,[100,100],[100])
        h2 =  self.decl_full_conn_layer("hidden2",h1,[100,100],[100])
        h3 =  self.decl_full_conn_layer("hidden3",h2,[100,100],[100])
        h4 =  self.decl_full_conn_layer("hidden4",h3,[100,100],[100])
        h5 =  self.decl_full_conn_layer("hidden5",h4,[100,100],[100])
        y =  self.decl_full_conn_layer("y",h5,[100,10],[10],isActive=False)
        loss = self.decl_softmax_corssentry_loss(y,y_)
        # return 1000,loss,x,y_,y,tf.train.AdamOptimizer(self.learn_rate)
        return 100,loss,x,y_,y,tf.train.GradientDescentOptimizer(self.learn_rate)


if __name__ == "__main__":
    print("#"*30)
    m = cifar.cifar("/home/lr/workspace/python/ai/data/cifar-10-batches-py/","/home/lr/workspace/python/ai/model/cifar_nn/model.ckpt")
    model = cifar_nn(m)
    model.plot(model.train(10,0.01),"count","loss","accuracy")

