#! /usr/bin/python
import tensorflow as tf
import numpy as np
from model_base import *
import cifar

class cifar_nn(model_base):
    
    def __init__(self,reader,batch=100):
        self.base =  super(cifar_nn,self)
        self.base.__init__(reader)
        self.batch=batch

    def decl_model(self):
        x = self.decl_placeholder("x",[None,3072])
        y_ = self.decl_placeholder("y_",[None,10])
        net = self.linear(x,100,is_batch_normal=True)
        net = self.linear(net,100,is_batch_normal=True)
        net = self.linear(net,100,is_batch_normal=True)
        net = self.linear(net,100,is_batch_normal=True)
        net = self.linear(net,100,is_batch_normal=True)
        net = self.linear(net,100,is_batch_normal=True)
        y = self.linear(net,10,is_active=False,is_batch_normal=True)
        loss = self.softmax_corssentry(y,y_)
        return self.batch,loss,x,y_,y,tf.train.AdamOptimizer(self.learn_rate)


if __name__ == "__main__":
    print("#"*30)
    m = config.cifar_train_reader('cifar_nn')
    model = cifar_nn(m, batch=100)
    model.plot(model.train(10, 0.001), "count", "loss", "accuracy")


