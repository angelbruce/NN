#! /usr/bin/python

import tensorflow as tf
import numpy as np
from cifar_check_base import *
import cifar

class cifar_cnn_test(cifar_check_base):
    
    def __init(self,reader):
        self.base = super(cifar_cnn_test,self)
        self.base.__init__(reader)

    def decl_predict(self):
        x = self.decl_placeholder("x",[None,3072])
        y_ = self.decl_placeholder("y_",[None,10])
        net = tf.reshape(x,[-1,32,32,3])
        for i in range(4):
           net = self.conv2d(net,[3,3],(i+1)*16,strides=2)
        net = self.conv2d(net,[3,3],10,strides=2,is_active=False)

        net = self.flat(net)

        y = tf.nn.softmax(net)

        return 1000,y,x,y_

if __name__ == "__main__":
    print("#"*30)
    m = config.cifar_test_reader('cifar_cnn')
    model = cifar_cnn_test(m)
    model.check()
