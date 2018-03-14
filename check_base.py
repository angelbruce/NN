import numpy as numpy
import tensorflow as tf
from model_base import *
import os
import matplotlib.pyplot as plt

class check_base(model_base):
    
    def __init__(self,reader):
        self.base = super(check_base,self)
        self.base.__init__(reader)

    def decl_predict(self):
        pass
    
    def check(self):
        with tf.Session() as sess:
            merge_op = tf.summary.merge_all()
            batch,y,x,y_ = self.decl_predict()
            tf.initialize_all_variables()
            saver = tf.train.Saver()
            segs = os.path.split(self.reader.checkpoints)
            ckpt = tf.train.get_checkpoint_state(segs[0])

            writer = tf.summary.FileWriter(segs[0],sess.graph)

            saver.restore(sess,ckpt.model_checkpoint_path)
            m = self.reader
            m.open()
            count = 0
            err_count = 0
            while m.has():
                batch_xs,batch_ys = m.next_datas(batch)
                py =  sess.run(y,feed_dict={x:batch_xs,y_:batch_ys})
                res = self.one_hot(py[0])
                count = count + 1
                if not self.equal_check(batch_ys[0],res):
                    err_count = err_count + 1
                else:
                    print("predict value is %s"%self.get_value(res))
            
            rate = (count-err_count)/(1.0*count)
            print("succ rate is %s "%rate)
            m.close()


    def get_value(self,a):
        for i in range(0,len(a)):
            if a[i] == 1.0: return i + 1
        return -1

    def one_hot(self,v):
        max_value = max(v)
        ret = [0.0 for i in range(0,len(v))]
        for i in range(0,len(v)):
            x = v[i]
            if x == max_value:
                ret[i] = 1
            else:
                ret[i] = 0
        return ret
            

    def equal_check(self,a,b):
        for i in range(0,len(a)):
            if a[i] == 1 and b[i] == 1:
                return True
            elif a[i] == 1 and b[i] != 1:
                #print(a,b)
                return False
            elif a[i] !=1 and b[i] == 1:
                #print(a,b)
                return False
        return False

