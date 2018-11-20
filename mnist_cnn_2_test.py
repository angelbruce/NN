#! /usr/bin/python

import tensorflow as tf
import numpy as np
from check_base import *
from config import *

class mnist_cnn_test_2(check_base):
    
    def __init(self,reader):
        self.base = super(mnist_cnn_test_2,self)
        self.base.__init__(reader)

    def decl_predict(self):
        x = self.placeholder("x",[None,784])
        y_ = self.placeholder("y_",[None,10])
        net = tf.reshape(x,[-1,28,28,1])
        net = self.conv2d(net,[3,3],32)
        net = self.sub_sample(net)
        net = self.conv2d(net,[3,3],64)
        net = self.sub_sample(net)
        net = self.flat(net)
        net = self.linear(net,512)
        net = self.linear(net,10,is_active=False)
        y = tf.nn.softmax(net)
        return 1,y,x,y_

if __name__ == "__main__":
    print("#"*30)
    m = config.mnist_test_reader('mnist_cnn_2')
    model = mnist_cnn_test_2(m)
    model.check()
