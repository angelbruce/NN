#! /usr/bin/python

import tensorflow as tf
import numpy as np
from check_base import *


class cifar_check_base(check_base):
    
    def __init(self,reader):
        self.base = super(cifar_check_base,self)
        self.base.__init__(reader)

    def get_value(self,label):
        k = 0 
        for i in range(len(label)):
            if label[i] == 1 :
                k = i
                break
        return self.reader.description(k)
