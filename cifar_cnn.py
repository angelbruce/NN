#!/usr/bin/python

import tensorflow as tf
import numpy as np
from model_base import *
import cifar


class cifar_cnn(model_base):

    def __init__(self, reader, batch=100):
        self.base = super(cifar_cnn, self)
        self.base.__init__(reader)
        self.batch = batch

    def decl_model(self):
        x = self.decl_placeholder("x", [None, 3072])
        y_ = self.decl_placeholder("y_", [None, 10])
        net = tf.reshape(x, [-1, 32, 32, 3])
        for i in range(4):
            net = self.conv2d(net, [3, 3], (i+1)*16, strides=2, is_batch_normal=True)
        net = self.conv2d(net, [3, 3], 10, strides=2, is_active=False, is_batch_normal=True)
        y = self.flat(net)
        loss = self.decl_softmax_corssentry_loss(y, y_)
        return self.batch, loss, x, y_, y, tf.train.AdamOptimizer(self.learn_rate)


if __name__ == "__main__":
    print("#"*30)
    m = config.cifar_train_reader('cifar_cnn')
    model = cifar_cnn(m, batch=100)
    model.train(1000, 0.001, save_mod=1)
