#!/usr/bin/python

import tensorflow as tf
import numpy as np
from model_base import *
import mnist

class mnist_cnn_1(model_base):

    def __init__(self,reader):
        self.base =  super(mnist_cnn_1,self)
        self.base.__init__(reader)

    def decl_model(self):
        batch = 100
        x = self.decl_placeholder("x",[None,784])
        y_ = self.decl_placeholder("y_",[None,10])
        input = tf.reshape(x,[-1,28,28,1])
        conv1 = tf.layers.conv2d(input,32,[5,5],padding='same',activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1,[2,2],2)
        conv2 = tf.layers.conv2d(pool1,64,[5,5],padding='same',activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2,[2,2],2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(dense,0.4,training=True)

        loss = self.decl_full_conn_softmax_crossentry_layer("fc",dropout,[1024,10],[10],y_)
       
        return batch,loss,x,y_

if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/train-images.idx3-ubyte","/home/lr/workspace/python/ai/data/train-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_cnn_1/model.ckpt")
    model = mnist_cnn_1(m)
    model.plot(model.train(1,0.001),"count","loss")
