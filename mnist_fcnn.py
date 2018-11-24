#!/usr/bin/python

import tensorflow as tf
import numpy as np
from model_base import *
import mnist
from config import *

class mnist_fcnn(model_base):

    def __init__(self,reader,batch=100):
        self.base =  super(mnist_fcnn,self)
        self.base.__init__(reader)
        self.batch=batch

    def decl_model(self):
        x = self.decl_placeholder("x",[None,784])
        y_ = self.decl_placeholder("y_",[None,10])
        x1 = tf.reshape(x,[-1,28,28,1])
        net = self.conv2d(x1,[3,3],32,is_batch_normal=True,strides=2,padding="V")
        net = self.conv2d(net,[2,2],32,is_batch_normal=True,strides=2,padding="V")
        net = self.conv2d(net,[3,3],32,is_batch_normal=True,strides=2,padding="V")
        net = self.conv2d(net,[2,2],10,is_batch_normal=True,strides=2,padding="V")

        z = self.flat(net)
        loss = self.softmax_corssentry(z,y_)

        return self.batch,loss,x,y_,z,None

if __name__ == "__main__":
    print("#"*30)
    m =  config.mnist_train_reader('mnist_fcnn')
    model = mnist_fcnn(m,batch=100)
    model.train(3,0.01)
