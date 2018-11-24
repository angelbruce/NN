#!/usr/bin/python

import tensorflow as tf
import numpy as np
from model_base import *
import mnist

class mnist_cnn(model_base):

    def __init__(self,reader,batch=100):
        self.base =  super(mnist_cnn,self)
        self.base.__init__(reader)
        self.batch=batch

    def decl_model(self):
        x = self.placeholder("x",[None,784])
        y_ = self.placeholder("y_",[None,10])
        x1 = tf.reshape(x,[-1,28,28,1])
        c1 = self.conv2d(x1,[5,5],32)
        p1 = self.sub_sample(c1)
        c2 = self.conv2d(p1,[5,5],64)
        p2 = self.sub_sample(c2)
        flat = self.flat(p2)
        dense = self.linear(flat,1024,keep_prob=0.4)
        y = self.linear(dense,10,is_active=False)
        loss = self.softmax_corssentry(y,y_)
        return self.batch,loss,x,y_,y,tf.train.AdamOptimizer(self.learn_rate)

if __name__ == "__main__":
    print("#"*30)
    m =  config.mnist_train_reader('mnist_cnn')
    model = mnist_cnn(m)
    model.plot(model.train(20,0.001),"count","loss","accuracy")