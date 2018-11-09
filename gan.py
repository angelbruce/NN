import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model_base import *
import mnist
import matplotlib.gridspec as gridspec
import os 

class GAN (nn_base):

    def __init__(self,reader,batch):
        self.base = super(GAN,self)
        self.reader = reader
        self.batch = batch


    def G_samples(self):
        return np.random.uniform(-1., 1., size=[self.batch, 100])

    def generator(self,z):
        W1 = self.decl_weight([100,128])
        B1 = self.decl_bias([128])
        Gh1 = tf.nn.relu(tf.matmul(z,W1) + B1)
        W2 = self.decl_weight([128,784])
        B2 = self.decl_bias([784])
        G_logit = tf.matmul(Gh1,W2) + B2
        G_prob = tf.nn.sigmoid(G_logit)

        return G_prob,[W1,B1,W2,B2]


    def descriminator(self,x):
        W1 = self.decl_weight([784,100])
        B1 = self.decl_bias([100])
        Dh1 = tf.nn.relu(tf.matmul(x,W1) + B1)
        W2 = self.decl_weight([100,1])
        B2 = self.decl_bias([1])
        D_logit = tf.matmul(Dh1,W2) + B2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob,D_logit,[W1,B1,W2,B2]


        

    def decl_model(self):
        D = self.decl_placeholder("var_d",[self.batch,784])
        d_real,d_logit_real,theta_D = self.descriminator(D)

        G = self.decl_placeholder("var_g",[self.batch,100])
        G_sample,theta_G= self.generator(G)
        d_fake,d_logit_fake,_= self.descriminator(G_sample)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real,labels=tf.ones_like(d_logit_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake,labels=tf.zeros_like(d_logit_fake)))


        d_loss = d_loss_real + d_loss_fake
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))


        d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=theta_D) 
        g_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=theta_G)

        return D,d_solver,d_loss,G,g_solver,g_loss,G_sample
    
    def train(self,epoch,out):
        self.epoch = epoch
        D,d_solver,d_loss,G,g_solver,g_loss,g_sample = self.decl_model()
        self.train_model(D,d_solver,d_loss,G,g_solver,g_loss,g_sample,out)

    def train_model(self,D,d_solver,d_loss,G,g_solver,g_loss,g_sample,out,save_mod=10):
        summary = []
        with tf.Session() as sess:
            m = self.reader
            print("initialize all variables")
            init = tf.global_variables_initializer()
            sess.run(init)
            count = 0
            e = 0
            while e < self.epoch:
                m.open()
                while m.has():
                    samples = self.G_samples()
                    batch_xs,batch_ys = m.next_datas(self.batch)
                    d_feed = {D:batch_xs,G:samples}
                    g_feed = {G:samples}
                    _, d_loss_curr = sess.run([d_solver, d_loss], feed_dict=d_feed)
                    _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict=g_feed) 
                    count = count + 1
                    print(count,d_loss_curr,g_loss_curr)

                    if count % save_mod == 0:
                        if not os.path.exists(out):
                            os.makedirs(out)
                        g_samples = sess.run([g_sample], feed_dict=g_feed) 
                        for i in range(0,self.batch):
                            s = g_samples[0][i]
                            fig = self.plot([s])
                            plt.savefig(out + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                            plt.close()

                m.close()
                e = e + 1

            return summary

    def plot(self,samples): 
        fig = plt.figure(figsize=(4, 4)) 
        gs = gridspec.GridSpec(4, 4) 
        gs.update(wspace=0.05, hspace=0.05) 
        for i, sample in enumerate(samples): 
            ax = plt.subplot(gs[i]) 
            plt.axis('off') 
            ax.set_xticklabels([]) 
            ax.set_yticklabels([]) 
            ax.set_aspect('equal') 
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r') 
        return fig

if __name__ == '__main__':
    print("#"*30)
    m = mnist.mnist("/home/lr/workspace/python/ai/data/train-images.idx3-ubyte","/home/lr/workspace/python/ai/data/train-labels.idx1-ubyte","/home/lr/workspace/python/ai/model/mnist_gan/model.ckpt")
    gan = GAN(m,100)
    gan.train(60000,"/home/lr/workspace/python/ai/model/mnist_gan/out")
