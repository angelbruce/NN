import numpy as np
import tensorflow as tf
from nn_base import *
import matplotlib.pyplot as plt
import os

class model_base(nn_base):

    def __init__(self,reader):
        self.base = super(model_base,self)
        self.base.__init__()
        self.reader = reader
    
    def decl_model(self):
        pass

    def train(self,epoch=1,learn_rate=0.001,log_dir=""):
        self.learn_rate = learn_rate
        batch,loss,x,y_,y,optimizer = self.decl_model()
        self.epoch = epoch
        summary = self.train_model(batch,loss,x,y_,y,optimizer)
        return summary

    def train_model(self,batch,loss,x_feed,y_feed,y,optimizer,save_mod=10):
        summary = []
        # train_steps = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(loss)
        if optimizer ==  None:
            optimizer = tf.train.RMSPropOptimizer(self.learn_rate)
        train_steps = optimizer.minimize(loss)
        with tf.Session() as sess:
            merge_op = tf.summary.merge_all()
            segs = os.path.split(self.reader.checkpoints)
            writer = tf.summary.FileWriter(segs[0],sess.graph)

            accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(y,1),tf.argmax(y_feed,1)), tf.float32))

            m = self.reader
            print("initialize all variables")
            init = tf.global_variables_initializer()
            sess.run(init)
            count = 0
            e = 0
            while e < self.epoch:
                m.open()
                while m.has():
                    batch_xs,batch_ys = m.next_datas(batch)
                    feed = {x_feed:batch_xs,y_feed:batch_ys}
                    _,l,acc,summary_str = sess.run([train_steps,loss,accuracy,merge_op],feed)
                    count = count + 1
                    print(count,l,acc)
                    summary.append({"count":count,"loss":l,"accuracy":acc})

                    if count % save_mod == 0 and m.checkpoints:
                        saver = tf.train.Saver()
                        saver.save(sess,m.checkpoints,count)
                        writer.add_summary(summary_str,count)

                m.close()
                e = e + 1

            return summary


    def plot(self,datas,x,*ylabels):
        plt.grid(True)
        xs = [k.get(x) for k in datas]
        r = len(ylabels)
        i = 1
        for y in ylabels:
            d = r*100 + 10 + i
            i = i + 1
            plt.subplot(d)
            ys = [k.get(y) for k in datas]
            plt.title(y)
            plt.plot(xs,ys)
        plt.show()