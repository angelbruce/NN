#! /usr/bin/python

import tensorflow as tf
import numpy as np
import mnist
from model_base import *

class mnist_lstm(model_base):

    def __init__(self,reader):
        self.base = super(mnist_lstm,self)
        self.base.__init__(reader)

    """
    mnist data is not sequence data which means the time spliter cannot affect the train results.
    but it may less the accuracy.
    found some other train datas to demonstrates the time relevance.
    """
    def decl_model(self):
        batch = 100        
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
        output = outputs[-1]                                        # current timestamp's output
        loss = self.decl_full_conn_softmax_crossentry_layer("lstm_out_hidden",output,[num_units,nclasses],[nclasses],y_)
        return batch,loss,x,y_,None

    

if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/train-images.idx3-ubyte","/home/lr/workspace/python/ai/data/train-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_lstm/model.ckpt")
    model = mnist_lstm(m)
    model.plot(model.train(1,0.001),"count","loss")
