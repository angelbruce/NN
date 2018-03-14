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
        c1 = self.decl_conv2d_layer("c1",x1,[3,3,1,2],[2])
        a1 = self.decl_avg_pool("ap1",c1)

        c2 = self.decl_conv2d_layer("c2",a1,[2,2,2,4],[4])
        a2 = self.decl_avg_pool("ap2",c2)

        c3 = self.decl_conv2d_layer("c2",a2,[3,3,4,6],[6])
        a3 = self.decl_avg_pool("ap3",c3)

        c4 = self.decl_conv2d_layer("c4",a3,[2,2,6,10],[10])
        a4 = self.decl_avg_pool("ap4",c4)

        z = tf.reshape(a4,[-1,10])
        p = tf.nn.softmax(z)
        return 1,p,x, y_


if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/t10k-images.idx3-ubyte","/home/lr/workspace/python/ai/data/t10k-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_fcnn/model.ckpt")
    model = mnist_fcnn_test(m)
    model.check()
