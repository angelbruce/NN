import numpy as np
import tensorflow as tf
from model_base import *
import os
import matplotlib.pyplot as plt


class check_base(model_base):

    def __init__(self, reader):
        self.base = super(check_base, self)
        self.base.__init__(reader)

    def decl_predict(self):
        pass

    def check(self):
        with tf.Session() as sess:
            batch, y, x, y_ = self.decl_predict()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            segs = os.path.split(self.reader.checkpoints)
            ckpt = tf.train.get_checkpoint_state(segs[0])
            saver.restore(sess, ckpt.model_checkpoint_path)
            m = self.reader
            m.open()
            is_training = self.is_training
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
            index = tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32)
            y_probs = tf.argmax(y, 1)
            avg_prob,y_indices,result = [],[],[]
            while m.has():
                batch_xs, batch_ys = m.next_datas(batch)
                acc, y_ps, y_index = sess.run([accuracy, y_probs,index], feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
                print(acc,y_ps,y_index)
                avg_prob.append(acc)
                result.append(y_ps)
                y_indices.append(y_index)

            p_avg =  np.mean(avg_prob)
            print(p_avg,result,y_indices)
            print(p_avg)
            m.close()
            datas = [{"count":i,"acc":avg_prob[i]} for i in range(len(result))]
            self.plot(datas,"count","acc")

            return p_avg,avg_prob,result,y_indices
