import tensorflow as tf
import numpy as np
import mnist

def get_value(a):
    for i in range(0,len(a)):
        if a[i] == 1.0: return i 
    return -1

m = mnist.mnist("/home/lr/workspace/python/ai/data/train-images.idx3-ubyte","/home/lr/workspace/python/ai/data/train-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_nn/model.ckpt")
m.open()
datas = m.next_datas(60000)
x_train,y_train = datas[0],datas[1]
x_train = x_train / 255.0
m.close()
y_train = np.array([get_value(y) for y in y_train])

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=[784]),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5,batch_size=32)
output = model.evaluate(x_train, y_train,batch_size=32)
print("the result is %s" %(output))

model.save('/home/lr/workspace/python/ai/nn/tfk/model/hello.1')
