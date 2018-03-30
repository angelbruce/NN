#! /usr/bin/python

import tensorflow as tf
import numpy as np
from check_base import *
import mnist

class mnist_lstm_test(check_base):
    
    def __init(self,reader):
        self.base = super(mnist_lstm_test,self)
        self.base.__init__(reader)

    def decl_predict(self):
        batch = 1        
        data_len = 784
        num_units = 128
        time_steps = 28
        nclasses = 10
        x = self.decl_placeholder("x",[None,data_len])
        y_ = self.decl_placeholder("y",[None,nclasses])
        x1 = tf.reshape(x,[-1,time_steps,data_len/time_steps])
        x2 = tf.unstack(x1,time_steps,1)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units,state_is_tuple=True)
        init_state =  lstm_cell.zero_state(batch,tf.float32)
        outputs,_ = tf.nn.static_rnn(lstm_cell,x2,initial_state=init_state)
        w1 = self.decl_weight([num_units,nclasses])
        b1 = self.decl_bias([nclasses])
        z =  tf.matmul(outputs[-1],w1) + b1
        y = tf.nn.softmax(z)
        return 1,y,x,y_

if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/t10k-images.idx3-ubyte","/home/lr/workspace/python/ai/data/t10k-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_lstm/model.ckpt")
    model = mnist_lstm_test(m)
    model.check()
