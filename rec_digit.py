#! /usr/bin/python
import tensorflow as tf
from mnist_nn_test import *
from data_reader import *
import os

class digit_recognize(mnist_nn_test)




if __name__ == "__main__":
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/t10k-images.idx3-ubyte","/home/lr/workspace/python/ai/data/t10k-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_nn/model.ckpt")
    model = mnist_nn_test(m)
    model.check()




class dir_file_reader(data_reader):

    def __init__(self,dir):
        self.dir = dir
        self.datas=[]


    def open(self):
        files = os.listdir(self.dir)
        for f in files:
            if os.path.isfile(f):
                fl = file(f,'rb')
            else:
                print("entry")


    def read_file(self,file_mame):
        


    def has(self):
        pass


    def next_datas(self,count):
        pass

    def close(self):
        pass

    def description(label):
        pass

    

    

