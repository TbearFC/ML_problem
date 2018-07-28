import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

class SeriesPredictor:
    def __init__(self,input_dim,seq_size,hidden_dim=10):
        self.input_dim=input_dim
        self.seq_size=seq_size
        self.hidden_dim=hidden_dim

        self.W_out=tf.Variable(tf.random_normal([hidden_dim,1]),name='W_out')
        self.b_out=tf.Variable(tf.random_normal([1]),name='b_out')
        self.x=tf.placeholder(tf.float32,[None,seq_size,input_dim])
        self.y=tf.placeholder(tf.float32,[None,seq_size])
        self.cost=tf.reduce_mean(tf.square(self.model()-self.y))
        self.train_op=tf.train.AdamOptimizer(0.01).minimize(self.cost)
        self.saver=tf.train.Saver()


    def model(self):
        cell=rnn.BasicLSTMCell(self.hidden_dim)
        outputs,states=tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples=tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0),[num_examples,1,1])
        out=tf.matmul(outputs,W_repeated)+self.b_out
        out=tf.squeeze(out)
        return out
    def train(self,train_x,train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                _,mse=sess.run([self.train_op,self.cost],feed_dict={self.x:train_x,self.y:train_y})
                if i%100==0:
                    print(i,mse)
            save_path=self.saver.save(sess,'/Users/guanbear/PycharmProjects/untitled/model.ckpt')
            print('Model is saved to {}'.format(save_path))
    def test(self,test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess,'/Users/guanbear/PycharmProjects/untitled/model.ckpt')
            output=sess.run(self.model(),feed_dict={self.x:test_x})
            tr_x_one = np.reshape(np.array(test_x), -1)
            tr_y_one = np.reshape(np.array(output), -1)
            plt.plot(tr_x_one, tr_y_one,'r')
            plt.show()

           # print(output)


if __name__=='__main__':
    predictor=SeriesPredictor(input_dim=1,seq_size=4)
    train_x=[[[1], [2], [3], [4]],
               [[5],[6], [7], [8]],
              [[9],[10],[11],[12]],
              [[14],[15],[16],[17]]]
    train_y=[[1,2,3,4],
             [5,6,7,8],
             [6,5,5,4],
             [3,2,2,1]]
    test_x = [[[1], [2], [3], [4]],
               [[5],[6], [7], [8]],
              [[9],[10],[11],[12]],
              [[14],[15],[16],[17]]]
    tr_x_one=np.reshape(np.array(train_x),-1)
    tr_y_one = np.reshape(np.array(train_y),-1)
    plt.plot(tr_x_one,tr_y_one,'.')

   # tx=np.reshape(train_x,[1,None])
   # ty=np.reshape(train_y,[1,None])
    predictor.train(train_x,train_y)
    predictor.test(test_x)

    plt.show()
    plt.hold


