from time import time
t_import = time()
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
t_train = time()
l_rate = 0.5
batchSz=100
fc_1 = 100
fc_2 = 10

W1 = tf.Variable(tf.random_normal([784, fc_1],stddev=.1))
b1 = tf.Variable(tf.random_normal([fc_1],stddev=.1))
W2 = tf.Variable(tf.random_normal([fc_1, fc_2],stddev=.1))
b2 = tf.Variable(tf.random_normal([fc_2],stddev=.1))


img = tf.placeholder(tf.float32, [batchSz,784])
ans = tf.placeholder(tf.float32, [batchSz, 10])

logits1 = tf.nn.relu(tf.matmul(img, W1) + b1) 		# fully connected 1 and ReLu
prbs = tf.nn.softmax(tf.matmul(logits1, W2) + b2)	# fully connected 2 and softmax
loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(l_rate).minimize(loss)
numCorrect= tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
	imgs, anss = mnist.train.next_batch(batchSz)
	sess.run(train, feed_dict={img: imgs, ans: anss})
	sumAcc=0

#-------------------------------------------------

t_test = time()
for i in range(1000):
	imgs, anss= mnist.test.next_batch(batchSz)
	sumAcc+=sess.run(accuracy, feed_dict={img: imgs, ans: anss})

#-------------------------------------------------
print 'Import time: {}, training time: {}, testing time: {}, total time: {}\n'.format(
	t_train - t_import, t_test - t_train, time() - t_test, time() - t_import)

print sumAcc/1000.0
