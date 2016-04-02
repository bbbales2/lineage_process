#%%

import os

os.chdir('/home/bbales2/lineage_process')

import pickle

f = open('toTrain.pkl')
labels, features = pickle.load(f)
f.close()
#%%
mean = numpy.mean(features, axis = 0)
#%%
i2l = list(enumerate(list(set(labels))))
l2i = dict([(l, i) for i, l in i2l])
#%%

import tensorflow as tf
sess = tf.InteractiveSession()

#%%

x_image = tf.placeholder(tf.float32, shape=[None, 16, 16, 9])
y_ = tf.placeholder(tf.float32, shape=[None, len(i2l)])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([3, 3, 9, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([16 * 16 * 9, 256])
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(x_image, [-1, 16 * 16 * 9])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([256, len(i2l)])
b_fc2 = bias_variable([len(i2l)])

y_log = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_conv = tf.nn.softmax(y_log)

yloss = -tf.reduce_sum(y_*tf.log(y_conv))

cross_entropy = yloss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%
def one(N, i):
    a = numpy.zeros(N)
    a[i] = 1.0
    return a


lref = numpy.array([one(len(i2l), l2i[l]) for l in labels])

#%%

sess.run(tf.initialize_all_variables())

batch_size = 1000

yls = []
for r in range(100):
    for i in range(0, len(labels), batch_size):
        xs = features[i : i + batch_size]
        ys = lref[i : i + batch_size]

        train, yl, acc = sess.run([train_step, yloss, accuracy], feed_dict = {x_image : xs, y_ : ys, keep_prob: 0.5})

        yls.append(yl)
        print acc

    plt.plot(yls)
    plt.show()
    print ''
#%%
acc = sess.run([accuracy], feed_dict = {x_image: features, y_: lref, keep_prob: 1.0})
print acc
#%%