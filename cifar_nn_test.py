#! /usr/bin/python

import tensorflow as tf
import numpy as np
from cifar_check_base import *
import cifar

class cifar_nn_test(cifar_check_base):
    
    def __init(self,reader):
        self.base = super(cifar_nn_test,self)
        self.base.__init__(reader)

    def decl_predict(self):
        batch = 1
        x = self.decl_placeholder("x",[None,3072])
        y_ = self.decl_placeholder("y_",[None,10])
        h0 =  self.decl_full_conn_layer("hidden0",x,[3072,100],[100])
        h1 =  self.decl_full_conn_layer("hidden1",h0,[100,100],[100])
        h2 =  self.decl_full_conn_layer("hidden2",h1,[100,100],[100])
        h3 =  self.decl_full_conn_layer("hidden3",h2,[100,100],[100])
        h4 =  self.decl_full_conn_layer("hidden4",h3,[100,100],[100])
        h5 =  self.decl_full_conn_layer("hidden5",h4,[100,100],[100])
        y =  self.decl_full_conn_layer("y",h5,[100,10],[10],isActive=False)
        loss = tf.nn.softmax(y)
        return 1,loss,x, y_

if __name__ == "__main__":
    print("#"*30)
    m = cifar.cifar("/home/lr/workspace/python/ai/data/cifar-10-batches-py/","/home/lr/workspace/python/ai/model/cifar_nn/model.ckpt",is_test=True)
    model = cifar_nn_test(m)
    model.check()
