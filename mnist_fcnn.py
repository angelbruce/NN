#!/usr/bin/python

import tensorflow as tf
import numpy as np
from model_base import *
import mnist

class mnist_fcnn(model_base):

    def __init__(self,reader):
        self.base =  super(mnist_fcnn,self)
        self.base.__init__(reader)

    def decl_model(self):
        batch = 100
        x = self.decl_placeholder("x",[None,784])
        y_ = self.decl_placeholder("y_",[None,10])
        x1 = tf.reshape(x,[batch,28,28,1])
        c1 = self.decl_conv2d_layer("c1",x1,[3,3,1,2],[2])
        a1 = self.decl_avg_pool("ap1",c1)
        c2 = self.decl_conv2d_layer("c2",a1,[2,2,2,4],[4])
        a2 = self.decl_avg_pool("ap2",c2)
        c3 = self.decl_conv2d_layer("c2",a2,[3,3,4,6],[6])
        a3 = self.decl_avg_pool("ap3",c3)
        c4 = self.decl_conv2d_layer("c4",a3,[2,2,6,10],[10])
        a4 = self.decl_avg_pool("ap4",c4)
        z = tf.reshape(a4,[-1,10])
        tf.summary.histogram("z",z)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z,labels=y_)) 
        tf.summary.scalar("loss",loss)

        return batch,loss,x,y_,None

if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/train-images.idx3-ubyte","/home/lr/workspace/python/ai/data/train-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_fcnn/model.ckpt")
    model = mnist_fcnn(m)
    model.plot(model.train(1,0.01),"count","loss")
