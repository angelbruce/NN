#!/usr/bin/python

import tensorflow as tf
import numpy as np
from model_base import *
import cifar

class cifar_cnn1(model_base):
    def __init__(self, reader, batch=100):
        self.base = super(cifar_cnn, self)
        self.base.__init__(reader)
        self.batch = batch

    def decl_model(self):
        x = self.placeholder("x", [None, 3072])
        y_ = self.placeholder("y_", [None, 10])
        net = tf.reshape(x, [-1, 32, 32, 3])
        net = self.branch(net)
        net = self.branch(net)
        net = self.branch(net)
        net = self.branch(net)
        net = self.flat(net)
        y = self.linear(net, 10, is_active=False)
        loss = self.softmax_corssentry(y, y_)
        return self.batch, loss, x, y_, y, tf.train.RMSPropOptimizer(
            self.learn_rate)

    def branch(self, net):
        nodes = [self.branch_layer(net, 0),self.branch_layer(net, 1),self.branch_layer(net, 2),self.branch_layer(net, 3),self.branch_layer(net, 8)]
        return tf.concat(nodes,3)

    def branch_layer(self, net, i):
        z = net
        if i == 0:
            z = self.conv2d(z, [1, 1], 8, is_batch_normal=False)
            z = self.conv2d(z, [1, 3], 8, is_batch_normal=False)
            z = self.conv2d(z, [3, 1], 8, is_batch_normal=False)
            z = self.conv2d(z, [3, 3], 8, is_batch_normal=False)
        if i == 1:
            z = self.conv2d(z, [1, 3], 8, is_batch_normal=False)
            z = self.conv2d(z, [3, 1], 8, is_batch_normal=False)
            z = self.conv2d(z, [3, 3], 8, is_batch_normal=False)
        if i == 2:
            z = self.conv2d(z, [1, 3], 8, is_batch_normal=False)
            z = self.conv2d(z, [3, 1], 8, is_batch_normal=False)
        if i == 3:
            z = self.conv2d(z, [1, 1], 8, is_batch_normal=False)
        else:
            z = self.conv2d(z, [3, 3], 8, is_batch_normal=False)

        z = self.sub_sample(z)
        return z


if __name__ == "__main__":
    print("#" * 30)
    m = config.cifar_train_reader('cifar_cnn1')
    model = cifar_cnn1(m, batch=100)
    model.plot(model.train(10, 0.001), "count", "loss", "accuracy")
