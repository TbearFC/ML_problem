import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

learn_rate=0.01
training_epochs=100

x_train=np.linspace(-1,1,101)
y_train=2*x_train+np.random.randn(*x_train.shape)*0.33
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

w=tf.Variable(0.0)
y_model=tf.multiply(X,w)
cost=tf.square(Y-y_model)

train_op=tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for epoch in range(training_epochs):
    for(x,y) in zip(x_train,y_train):
        sess.run(train_op,feed_dict={X:x,Y:y})

W_val=sess.run(w)
sess.close()
plt.scatter(x_train,y_train)
y_new=x_train*W_val
plt.plot(x_train,y_new,'r')
plt.show()