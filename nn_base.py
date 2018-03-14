import numpy as np
import tensorflow as tf

class nn_base(object):
  
    def decl_placeholder(self,name,shape):
        v = tf.placeholder(tf.float32,shape,name)
        return v
    
    def decl_weight(self,shape):
        init = tf.truncated_normal(shape,mean=0.0, stddev=0.1)
        weight = tf.Variable(init,name="W")
        tf.summary.histogram("W",weight)
        return weight

    def decl_bias(self,shape):
        init = tf.constant(1,tf.float32,shape)
        bias = tf.Variable(init,name="b")
        tf.summary.histogram("b",bias)
        return bias

    def decl_conv2d_layer_with_maxpool(self,name,x,weight_shape,bias_shape,strides=[1,1,1,1],padding='SAME'):
        with tf.name_scope(name):
            w = self.decl_weight(weight_shape)
            b = self.decl_bias(bias_shape)
            z = tf.nn.conv2d(x,w,strides,padding) + b
            y = tf.nn.relu(z)
            m = tf.nn.max_pool(y,ksize=[1,2,2,1],strides=[1,1,1,1],padding = 'SAME')

            tf.summary.histogram(name + "/conv2d",z)
            tf.summary.histogram(name + "/relu",y)
            tf.summary.histogram(name + "/max_pool",m)

            return m

    def decl_conv2d_layer(self,name,x,weight_shape,bias_shape,strides=[1,1,1,1],padding='SAME'):
        with tf.name_scope(name):
            w = self.decl_weight(weight_shape)
            b = self.decl_bias(bias_shape)
            z = tf.nn.conv2d(x,w,strides,padding) + b
            y = tf.nn.relu(z)

            tf.summary.histogram(name + "/conv2d",z)
            tf.summary.histogram(name + "/relu",y)

            return y


    def decl_full_conn_layer(self,name,x,weight_shape,bias_shape):
        with tf.name_scope(name):
            w = self.decl_weight(weight_shape)
            b = self.decl_bias(bias_shape)
            z = tf.matmul(x,w) + b
            y = tf.nn.relu(z)

            tf.summary.histogram(name + "/matmul",z)
            tf.summary.histogram(name + "/relu",y)

            return y

    def decl_full_conn_softmax_layer(self,name,x,weight_shape,bias_shape):
        with tf.name_scope(name):
            w = self.decl_weight(weight_shape)
            b = self.decl_bias(bias_shape)
            z = tf.matmul(x,w) + b
            y = tf.nn.softmax(z)

            tf.summary.histogram(name + "/matmul",z)
            tf.summary.histogram(name + "/softmax",y)

            return y

    def decl_full_conn_softmax_crossentry_layer(self,name,x,weight_shape,bias_shape,y_):
        with tf.name_scope(name):
            w = self.decl_weight(weight_shape)
            b = self.decl_bias(bias_shape)
            z = tf.matmul(x,w) + b
            per_loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y_,name=name+"_per_crossentry")
            loss = tf.reduce_mean(per_loss)

            tf.summary.histogram(name + "/matmul",z)
            tf.summary.histogram(name + "/per_loss",per_loss)
            tf.summary.histogram(name + "/loss",loss)


            return loss

    def decl_avg_pool(self,name,x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID'):
        with tf.name_scope(name):
            ap = tf.nn.avg_pool(x,ksize,strides,padding)
            tf.summary.histogram(name + "/avg_pool",ap)
            return ap       

    def decl_max_pool(self,name,x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID'):
        with tf.name_scope(name):
            ap = tf.nn.max_pool(x,ksize,strides,padding)
            tf.summary.histogram(name + "/max_pool",ap)
            return ap            
     

    def decl_drop_out_layer(self,x, keep_prob, name):
        dp = tf.nn.dropout(x,keep_prob,name=name)
        tf.summary.histogram(name + "/drop_out",dp)
        return dp


    def decl_dense_layer(self,name,x,units,activation=tf.nn.relu,use_bias=True):
        dl = tf.layers.dense(x,units,activation,use_bias,name=name)
        tf.summary.histogram(name + "/dense",dl)
        return dl