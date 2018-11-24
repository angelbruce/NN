#! /usr/bin/python

import tensorflow as tf
import numpy as np
from check_base import *
import mnist

class mnist_fcnn_test(check_base):
    
    def __init(self,reader):
        self.base = super(mnist_fcnn_test,self)
        self.base.__init__(reader)

    def decl_predict(self):
        x = self.decl_placeholder("x",[None,784])
        y_ = self.decl_placeholder("y_",[None,10])
        x1 = tf.reshape(x,[-1,28,28,1])
        net = self.conv2d(x1,[3,3],32,is_batch_normal=True,strides=2,padding="V")
        # net = self.sub_sample(net,pool_op=tf.nn.avg_pool)
        net = self.conv2d(net,[2,2],32,is_batch_normal=True,strides=2,padding="V")
        # net = self.sub_sample(net,pool_op=tf.nn.avg_pool)
        net = self.conv2d(net,[3,3],32,is_batch_normal=True,strides=2,padding="V")
        # net = self.sub_sample(net,pool_op=tf.nn.avg_pool)
        net = self.conv2d(net,[2,2],10,is_batch_normal=True,strides=2,padding="V")
        # net = self.sub_sample(net,pool_op=tf.nn.avg_pool)
        z = self.flat(net)
        p = tf.nn.softmax(z)
        return 1000,p,x, y_


if __name__ == "__main__":
    print("#"*30)
    m =  config.mnist_test_reader('mnist_fcnn')
    model = mnist_fcnn_test(m)
    model.check()
