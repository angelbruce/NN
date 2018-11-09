#! /usr/bin/python
import numpy as np

array = np.loadtxt('/home/lr/data/a')
datas = np.reshape(array,[-1,784])
print(datas)
print(datas.shape)