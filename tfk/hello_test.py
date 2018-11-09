import tensorflow as tf
mnist = tf.keras.datasets.mnist

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

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.load_model("/home/lr/workspace/python/ai/nn/tfk/model/hello.")
ret = model.predict(x_test)

corr  = 0
for i in range(0,len(ret)):
    reti = ret[i]
    pred = get_value(one_hot(reti))
    if pred == y_test[i]:
        corr += 1.0

print(corr / len(ret))




