import numpy as np
import tensorflow as tf
from config import *

class nn_base(object):
    
    def __init__(self):
        self.is_training = tf.placeholder(tf.bool)
        
  
    def decl_placeholder(self,name,shape):
        v = tf.placeholder(tf.float32,shape,name)
        return v
    
    def decl_Var(self,name,shape,mean=0.0,stdv=0.1): 
        init = tf.truncated_normal(shape,mean=0.0,stddev=0.1)
        var = tf.Variable(init,name=name)
        tf.summary.histogram(name,var)

        return var

    def decl_weight(self,shape,mean=0.0,stddev=0.1):
        init = tf.truncated_normal(shape,mean=mean, stddev=stddev)
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

            tf.summary.histogram("conv2d",z)
            tf.summary.histogram("relu",y)
            tf.summary.histogram("max_pool",m)

            return m

    def decl_conv2d_layer(self,name,x,weight_shape,bias_shape,strides=[1,1,1,1],padding='SAME'):
        with tf.name_scope(name):
            w = self.decl_weight(weight_shape)
            b = self.decl_bias(bias_shape)
            z = tf.nn.bias_add(tf.nn.conv2d(x,w,strides,padding,use_cudnn_on_gpu=config.use_cudnn_on_gpu), b)
            y = tf.nn.relu(z)

            tf.summary.histogram("conv2d",z)
            tf.summary.histogram("relu",y)

            return y
        

    def decl_full_conn_layer(self,name,x,weight_shape,bias_shape,isActive=True):
        with tf.name_scope(name):
            w = self.decl_weight(weight_shape)
            b = self.decl_bias(bias_shape)
            z = tf.matmul(x,w) + b
            tf.summary.histogram("matmul",z)

            if not isActive :
                return z

            else:
                y = tf.nn.relu(z)
                tf.summary.histogram("relu",y)

                return y

    def decl_full_conn_softmax_layer(self,name,x,weight_shape,bias_shape):
        with tf.name_scope(name):
            w = self.decl_weight(weight_shape)
            b = self.decl_bias(bias_shape)
            z = tf.matmul(x,w) + b
            y = tf.nn.softmax(z)

            tf.summary.histogram("matmul",z)
            tf.summary.histogram("softmax",y)

            return y

    def decl_full_conn_softmax_crossentry_layer(self,name,x,weight_shape,bias_shape,y_):
        with tf.name_scope(name):
            w = self.decl_weight(weight_shape)
            b = self.decl_bias(bias_shape)
            z = tf.matmul(x,w) + b
            per_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y_,name=name+"_per_crossentry")
            loss = tf.reduce_mean(per_loss)

            tf.summary.histogram("matmul",z)
            tf.summary.histogram("per_loss",per_loss)
            tf.summary.histogram("loss",loss)


            return loss

    def decl_softmax_corssentry_loss(self,logits,labels) :
        per_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels,name="per_loss")
        loss = tf.reduce_mean(per_loss)

        tf.summary.histogram("per_loss",per_loss)
        tf.summary.histogram("loss",loss)

        return loss

    def decl_avg_pool(self,name,x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID'):
        with tf.name_scope(name):
            ap = tf.nn.avg_pool(x,ksize,strides,padding)
            tf.summary.histogram("avg_pool",ap)
            return ap       

    def decl_max_pool(self,name,x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID'):
        with tf.name_scope(name):
            ap = tf.nn.max_pool(x,ksize,strides,padding)
            tf.summary.histogram("max_pool",ap)
            return ap            
     

    def decl_drop_out_layer(self,x, keep_prob, name):
        dp = tf.nn.dropout(x,keep_prob,name=name)
        tf.summary.histogram("drop_out",dp)
        return dp


    def decl_dense_layer(self,name,x,units,activation=tf.nn.relu,use_bias=True):
        dl = tf.layers.dense(x,units,activation,use_bias,name=name)
        tf.summary.histogram("dense",dl)
        return dl


    def placeholder(self,name,shape):
        return self.decl_placeholder(name,shape)

    def conv2d(self,x,kernel,outdim,name="conv2d",strides=1,padding='SAME',weight_mean=0.0,weight_stddev=0.1,is_batch_normal=False,is_active=True,is_all=False,active_op=tf.nn.relu,is_dropout=False,keep_prob=0.5):
        if padding == 'S' : padding = 'SAME'
        elif padding == 'V' : padding = 'VALID'
        strides = [1,strides,strides,1]
        name = name + "_" + str(kernel[0])+"x"+str(kernel[1])+"x"+str(outdim)
        with tf.name_scope(name):
            weight_shape = [kernel[0],kernel[1],x.shape[-1].value,outdim]
            bias_shape = [outdim]
            w = self.decl_weight(weight_shape,mean=weight_mean,stddev=weight_stddev)
            b = self.decl_bias(bias_shape)
            z = tf.nn.bias_add(tf.nn.conv2d(x,w,strides,padding) , b)
            tf.summary.histogram("conv2d",z)
            if is_dropout:
                z = tf.nn.dropout(z,keep_prob)
                tf.summary.histogram("dropout",z)

            if is_batch_normal:
              
                # z = tf.layers.batch_normalization(z,training=self.is_training)
                z = self.batch_normalization(z,outdim)
                tf.summary.histogram("batch_normal",z)

            if is_active:
                y = active_op(z)
                tf.summary.histogram("active_op",y)

                if is_all: return y,z,w,b
                else: return y

            else:
                if is_all: return z,z,w,b
                else: return z                               

            return m


    def batch_normalization(self,layer,outdim):
        gamma = tf.Variable(tf.ones([outdim]))
        beta = tf.Variable(tf.zeros([outdim]))
        pop_mean = tf.Variable(tf.zeros([outdim]), trainable=False)
        pop_variance = tf.Variable(tf.ones([outdim]), trainable=False)
        epsilon = 1e-3

        def batch_norm_training():
            axes = list(range(len(layer.shape)-1))
            batch_mean, batch_variance = tf.nn.moments(layer,axes, keep_dims=False)
            decay = 0.99
            train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))
            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(layer, batch_mean, batch_variance, beta, gamma, epsilon)

        def batch_norm_inference():
            return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon)

        batch_normalized_output = tf.cond(self.is_training, batch_norm_training, batch_norm_inference)

        return batch_normalized_output


    def sub_sample(self,x,name='sub_sample',strides=2,padding='VALID',pool_op=tf.nn.max_pool):
        if padding == 'S' : padding = 'SAME'
        elif padding == 'V' : padding = 'VALID'
        strides = [1,strides,strides,1]
        ksize=[1,strides,strides,1]

        if pool_op == tf.nn.max_pool: name=name + '_max_pool'
        elif pool_op == tf.nn.avg_pool: name=name + '_avg_pool'
            
        with tf.name_scope(name):
            x = pool_op(x,ksize,strides,padding)
            tf.summary.histogram("sub_sample",x)

        return x


    def linear(self,x,outdim,name="linear",weight_mean=0.0,weight_stddev=0.1,is_batch_normal=False,is_active=True,is_all=False,active_op=tf.nn.relu,is_dropout=False,keep_prob=0.5):
        name = name + "_" + str(x.shape[-1].value) + "x" + str(outdim) 
        with tf.name_scope(name):
            weight_shape = [x.shape[-1].value,outdim]
            bias_shape = [outdim]
            w = self.decl_weight(weight_shape,mean=weight_mean,stddev=weight_stddev)
            b = self.decl_bias(bias_shape)
            z = tf.nn.bias_add(tf.matmul(x,w) , b)
            tf.summary.histogram("matmul",z)
            if is_dropout:
                z = tf.nn.dropout(z,keep_prob)
                tf.summary.histogram("dropout",z)

            if is_batch_normal:
                z = self.batch_normalization(z,outdim)
                tf.summary.histogram("batch_normal",z)

            if is_active:
                y = active_op(z)
                tf.summary.histogram("active_op",y)

                if is_all: return y,z,w,b
                else: return y

            else:
                if is_all: return z,z,w,b
                else: return z                               

            return m


    def flat(self,x,name="flatten"):
        dim = 1
        for i,idim in enumerate(x.shape):
            if i == 0 : continue
            dim = dim * idim
    
        x = tf.reshape(x,[-1,dim])
        tf.summary.histogram("flatten",x)

        return x
        
    def softmax_corssentry(self,logits,labels) :
        return self.decl_softmax_corssentry_loss(logits,labels)


    def conv2d_branch(self,x,branch_kernels,outdim,strides=1,padding='S',weight_mean=0.0,weight_stddev=0.1,is_batch_normal=False,is_active=True,active_op=tf.nn.relu,is_dropout=False,keep_prob=0.5,is_resnet=False):
        nets = []
        if is_resnet: nets.append(x)
        for kernels in branch_kernels:
            z = x
            for kernel in kernels:
                z = self.conv2d(z,kernel,outdim,padding=padding,strides=strides,weight_mean=weight_mean,weight_stddev=weight_stddev,is_batch_normal=is_batch_normal,is_active=is_active,active_op=active_op,is_dropout=is_dropout,keep_prob=keep_prob)
            nets.append(z)

        net = nets[0]        
        for i,n in enumerate(nets):
            if i==0: continue
            net = tf.concat(net,n,3)

        return net

            