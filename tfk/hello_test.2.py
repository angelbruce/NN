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


m = mnist.mnist("/home/lr/workspace/python/ai/data/t10k-images.idx3-ubyte","/home/lr/workspace/python/ai/data/t10k-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_nn/model.ckpt")
m.open()
datas = m.next_datas(10000)
m.close()
x_test,y_test = datas[0],datas[1]
x_test = x_test / 255.0
y_test = np.array([get_value(y) for y in y_test])

model = tf.keras.models.load_model("/home/lr/workspace/python/ai/nn/tfk/model/hello.1")
ret = model.predict(x_test)

corr  = 0
for i in range(0,len(ret)):
    reti = ret[i]
    pred = get_value(one_hot(reti))
    if pred == y_test[i]:
        corr += 1.0

print(corr / len(ret))




