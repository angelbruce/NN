import tensorflow as tf
import numpy as np
import cifar

def get_value(a):
    for i in range(0,len(a)):
        if a[i] == 1.0: return i 
    return -1

m = cifar.cifar("/home/lr/workspace/python/ai/data/cifar-10-batches-py/","/home/lr/workspace/python/ai/model/cifar_cnn/model.ckpt")
m.open()
datas = m.next_datas(50000)
x_train,y_train = datas[0],datas[1]
print(x_train.shape)
print(y_train.shape)
x_train = x_train / 255.0
m.close()
y_train = np.array([get_value(y) for y in y_train])

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=[3072]),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dense(32, activation=tf.nn.relu),
  tf.keras.layers.Dense(16, activation=tf.nn.sigmoid),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=1000,batch_size=100)
output = model.evaluate(x_train, y_train,batch_size=100)
print("the result is %s" %(output))

model.save('/home/lr/workspace/python/ai/nn/tfk/model/hello.2')
