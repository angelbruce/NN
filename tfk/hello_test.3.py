import tensorflow as tf
import numpy as np
import mnist

def get_value(a):
    for i in range(0,len(a)):
        if a[i] == 1.0: return i 
    return -1


def one_hot(v):
    max_value = max(v)
    ret = [0.0 for i in range(0,len(v))]
    for i in range(0,len(v)):
        x = v[i]
        if x == max_value:
            ret[i] = 1
        else:
            ret[i] = 0
    return ret



array = np.loadtxt('/home/lr/data/a')
x_test = np.reshape(array,[-1,784])
y_test = np.zeros([1,10])
x_test = x_test / 255.0

model = tf.keras.models.load_model("/home/lr/workspace/python/ai/nn/tfk/model/hello.1")
ret = model.predict(x_test)

vec = one_hot(ret[0])
print(vec)




