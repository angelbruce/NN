#!/usr/bin/python

import tensorflow as tf
import numpy as np
from model_base import *
from config import *

class mnist_cnn_2(model_base):

    def __init__(self,reader,batch=100):
        self.base =  super(mnist_cnn_2,self)
        self.base.__init__(reader)
        self.batch=batch

    def decl_model(self):
        x = self.placeholder("x",[None,784])
        y_ = self.placeholder("y_",[None,10])
        net = tf.reshape(x,[-1,28,28,1])
        net = self.conv2d(net,[3,3],32,is_batch_normal=True)
        net = self.sub_sample(net)
        net = self.conv2d(net,[3,3],64,is_batch_normal=True)
        net = self.sub_sample(net)
        net = self.flat(net)
        net = self.linear(net,512)
        y = self.linear(net,10,is_active=False,is_batch_normal=True)

        loss = self.softmax_corssentry(y,y_)
        return self.batch,loss,x,y_,y,tf.train.AdamOptimizer()

if __name__ == "__main__":
    print("#"*30)
    m =  config.mnist_train_reader('mnist_cnn_2')
    model = mnist_cnn_2(m)
    model.plot(model.train(1,0.001),"count","loss")