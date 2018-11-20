#! /usr/bin/python

import tensorflow as tf
import numpy as np
from cifar_check_base import *
import cifar

class cifar_nn_test(cifar_check_base):
    
    def __init(self,reader,batch=1):
        self.base = super(cifar_nn_test,self)
        self.base.__init__(reader)
        self.batch=1

    def decl_predict(self):
        x = self.placeholder("x",[None,3072])
        y_ = self.placeholder("y_",[None,10])
        net = self.linear(x,100)
        net = self.linear(net,100)
        net = self.linear(net,100)
        net = self.linear(net,100)
        net = self.linear(net,100)
        net = self.linear(net,100)
        net = self.linear(net,10,is_active=False)
        y = tf.nn.softmax(net)
        
        return self.batch,y,x,y_

if __name__ == "__main__":
    print("#"*30)
    m = cifar.cifar("/home/lr/workspace/python/ai/data/cifar-10-batches-py/","/home/lr/workspace/python/ai/model/cifar_nn/model.ckpt",is_test=True)
    model = cifar_nn_test(m)
    model.check()
