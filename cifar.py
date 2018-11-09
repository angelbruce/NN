#! /usr/bin/python

import cPickle as pickle
import os

import numpy as np

from data_reader import *


class cifar(data_reader):
    BATCH_FILE_NAME = 'batches.meta'
    BATCH_DATA_PFX = 'data_batch_'
    TEST_DATA_NAME = 'test_batch'
    
    def __init__(self,dir,checkpoints,is_test = False):
        self.checkpoints = checkpoints
        self.is_test = is_test
        self.batch = 5
        self.idx_cursor = 0
        self.total = 0
        self.idx = 0
        self.dir =  dir
        self.datas = []
        self.labels = []
        self.idx_cursor = 0
        self.load_description()

    def open(self):
        self.pre_load_check(10000)


    def has(self):
        if not self.is_test:
            return self.idx  <= self.batch and self.idx_cursor < self.total
        else:
            return self.idx_cursor < self.total

    def batch_has(self):
        return self.idx < self.batch 


    def next_datas(self,count):
        edx = self.idx_cursor + count 
        if not self.is_test:
            self.pre_load_check(count)

        if edx > self.total: edx = self.total
        imgs = self.datas[self.idx_cursor:edx ]
        labels = self.labels[self.idx_cursor:edx ]
        # print(self.idx_cursor,edx,self.total)
        self.idx_cursor = edx

        for i in range(len (imgs) ):
            imgs[i]  = imgs[i ] / 256.0
      
        return (imgs,labels)

    def pre_load_check(self,count):
        if not self.is_test:
            idx = self.idx
            while (self.idx_cursor + count) >= self.total:
                if self.batch_has():
                    self.load_data(idx + 1)
                    idx = idx + 1
                else:
                    break
        else:
            self.load_test_data()
     
    def load_data(self,idx):
        self.idx = idx
        filename = os.path.join(self.dir,cifar.BATCH_DATA_PFX+str(idx))
        # print(filename)
        datas = self.load(filename)
        imgs = datas['data']
        labels = datas['labels']
        one_hots = [self.label_one_hot(k) for k in labels]
        self.datas.extend(imgs)
        self.labels.extend(one_hots)
        self.total = len(self.datas)

    def load_test_data(self):
        filename = os.path.join(self.dir,cifar.TEST_DATA_NAME)
        datas = self.load(filename)
        imgs = datas['data']
        labels = datas['labels']
        one_hots = [self.label_one_hot(k) for k in labels]
        self.datas.extend(imgs)
        self.labels.extend(one_hots)
        self.total = len(self.datas)

        
    def close(self):
        self.idx_cursor = 0

    def load(self,filename):
        f = file(filename,'rb')
        return pickle.load(f)    

    def description(self,label):
        return self.label_descs[label]


    def load_description(self):
        desc_file = os.path.join(self.dir,cifar.BATCH_FILE_NAME)
        alls = self.load(desc_file)
        self.label_descs = alls['label_names']
        self.type_len = len(self.label_descs)

    def label_one_hot(self,label):
        vec = [0 for k in range(self.type_len)]
        vec[label] = 1
        # print(label,vec)
        return np.array(vec,np.float32)

        


if __name__ == '__main__':
    cf = cifar('/home/lr/workspace/python/ai/data/cifar-10-batches-py/','')
    cf.open()
    # while cf.has():
    #     imgs,labels =  cf.next_datas(1000)
    #     print(len(imgs),len(labels),len(imgs[0]),len(imgs[len(imgs)-1]),len(labels[0]))


    print("#"*30)
    cf = cifar('/home/lr/workspace/python/ai/data/cifar-10-batches-py/','',is_test=True)
    cf.open()
   
    imgs,labels =  cf.next_datas(1)
    print(imgs)
    print(len(imgs),len(labels),len(imgs[0]),len(imgs[len(imgs)-1]),len(labels[0]))
