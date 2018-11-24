#! /usr/bin/python
import tensorflow as tf
import numpy as np
from model_base import *
import mnist
from config import *

class mnist_nn(model_base):
    
    def __init__(self,reader,batch=100):
        self.base = super(mnist_nn,self)
        self.base.__init__(reader)
        self.batch=batch

    def decl_model(self):
        x = self.placeholder("x",[None,784])
        y_ = self.placeholder("y_",[None,10])
        net = self.linear(x,100,is_batch_normal=True)
        y = self.linear(net,10,is_active=False)
        loss = self.softmax_corssentry(y,y_)
        return self.batch,loss,x,y_,y,tf.train.AdamOptimizer(self.learn_rate)


if __name__ == "__main__":
    print("#"*30)
    m = config.mnist_train_reader('mnist_nn')
    model = mnist_nn(m)
    model.plot(model.train(1),"count","loss","accuracy")