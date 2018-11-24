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
        x = self.placeholder("x",[None,784])
        y_ = self.placeholder("y_",[None,10])
        x1 = tf.reshape(x,[-1,28,28,1])
        c1 = self.conv2d(x1,[5,5],32)
        p1 = self.sub_sample(c1)
        c2 = self.conv2d(p1,[5,5],64)
        p1 = self.sub_sample(c2)
        flat = self.flat(p1)
        dense = self.linear(flat,1024)
        net = self.linear(dense,10)
        y = tf.nn.softmax(net)
        return 1,y,x,y_

if __name__ == "__main__":
    print("#"*30)
    m =  config.mnist_test_reader('mnist_cnn')
    model = mnist_cnn_test(m)
    model.check()
