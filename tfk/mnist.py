#! /usr/bin/python
import numpy as np
import tensorflow as tf
from data_reader import *

class mnist(data_reader):
        """ mnist data reader """

        def __init__(self,imgs,labels,checkpoints):
            self.imgs = imgs
            self.labels = labels
            self.checkpoints = checkpoints
            self.cursor = 0
            self.length = 0
        
        def open(self):
            self.f_img = open(self.imgs,'rb')
            self.f_lbl = open(self.labels,'rb')
            self.load_image_head()
            self.load_label_head()

        
        def load_image_head(self):
            f = self.f_img
            msb = self.read_num (f,4)
            self.length =  self.read_num(f,4)
            rows = self.read_num(f,4)
            cols = self.read_num(f,4)
            self.img_per_len =  rows * cols
            print("images has %d items"%self.length)
        
        def load_label_head(self):
            f = self.f_lbl
            msb = self.read_num(f,4)
            l = self.read_num(f,4)
            print("labels has %d items"%l)
        
        def has(self):
            if self.cursor < (self.length-1): return 1
            else: return 0

        def next_datas(self,count):
            if not self.has(): return (None,None)
            plen = self.img_per_len
            image_items = np.zeros([count,plen])
            for i in range(0,count):
                image_items[i] = [1.0*self.read_num(self.f_img,1) for x in range(0,plen)]

            label_items =  np.zeros([count,10])
            for i in range (0,count):
                l = self.read_num(self.f_lbl,1)
                label_items[i][l] = 1.0

            self.cursor += count
            datas = (np.array(image_items,np.float32),np.array(label_items,np.float32))

            return datas

        
        def close(self):
            self.f_img.close()
            self.f_lbl.close()
            self.cursor = 0
            
    
        def read_num(self,f,n):
            v = int(0)
            bs = f.read(n)
            for b in bs:
                v = v << 8 | ord(b)
            return v




if __name__ == '__main__':
    print("#"*30)
    m = mnist("/home/lr/workspace/python/ai/data/t10k-images.idx3-ubyte","/home/lr/workspace/python/ai/data/t10k-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_nn_cos/model.ckpt")
    m.open()
    print(m.has())
    m.close()