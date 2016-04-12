#%%
import os
import re
import numpy
import matplotlib
import json
import itertools
import time
import skimage.io

os.chdir('/home/bbales2/lineage_process')

#%%

testClasses = sorted(os.listdir('filtered_sprites_160x160'))

#%%

images = {}

for i, code in enumerate(testClasses):
    folder = 'all_sprites/{0}'.format(code)
    images[folder] = os.listdir(folder)

f = open('images2.json', 'w')
json.dump(images, f)
f.close()

#%%

f = open('reference.json')
reference = json.load(f)
f.close()

classes = []

for ref in reference:
    if reference[ref] == 'true':
        classes.append(ref.split('/')[1])

numpy.random.shuffle(classes)

#classes = classes[:100]

classes = sorted(classes)

#%%
backgrounds = []

for f in os.listdir('screencast_frames'):
    backgrounds.append(skimage.io.imread('screencast_frames/{0}'.format(f))[40:360, :300])
#%%

dframes = {}

labels = []
numImages = 0
rpts = 16
for i, code in enumerate(classes):
    f = open('all_sprites_160x160_index/{0}.index'.format(code))
    data = json.load(f)
    f.close()

    images = numpy.load('filtered_sprites_160x160/{0}'.format(code))

    numImages += rpts * images.shape[0]

    labels.extend([code] * rpts * images.shape[0])

    print i
#%%
features = numpy.memmap('/media/bbales2/189304c6-f6e1-4da9-b130-6fc829d5544c/tmp.npy', dtype = 'float32', mode = 'w+', shape = (numImages, 128, 128, 3))

idx = 0
for i, code in enumerate(classes):
    f = open('all_sprites_160x160_index/{0}.index'.format(code))
    data = json.load(f)
    f.close()

    images = numpy.load('filtered_sprites_160x160/{0}'.format(code))

    for im in images:
        im2 = im[80 - 64 : 80 + 64, 80 - 64 : 80 + 64, :4]
        mask = (im2[:, :, 3] > 254.5)
        for r in range(rpts):
            background = backgrounds[numpy.random.randint(0, len(backgrounds))]

            ii = numpy.random.randint(0, background.shape[0] - 128)
            jj = numpy.random.randint(0, background.shape[1] - 128)

            backgroundSample = background[ii : ii + 128, jj : jj + 128]

            im3 = numpy.array(im2)
            im3[:, :, 0] += mask * backgroundSample[:, :, 0]
            im3[:, :, 1] += mask * backgroundSample[:, :, 1]
            im3[:, :, 2] += mask * backgroundSample[:, :, 2]
            im4 = im3[:, :, :3] / 255.0
            #dy, dx, da = numpy.gradient(im4)
            #gradient = numpy.max(dx**2 + dy**2, axis = 2)
            features[idx, :, :, :3] = im4
            #features[idx, :, :, 3] = gradient
            #plt.imshow(gradient)
            #plt.show()
            #plt.imshow(im4)
            #plt.show()
            #print 'hi'
            idx += 1

    print i

del features
#%%
features = numpy.memmap('/media/bbales2/189304c6-f6e1-4da9-b130-6fc829d5544c/tmp.npy', dtype = 'float32', mode = 'r', shape = (numImages, 128, 128, 3))

#features = numpy.array([(feature - mean) for feature in features])
i2l = list(enumerate(list(set(labels))))
l2i = dict([(l, i) for i, l in i2l])

def one(N, i):
    a = numpy.zeros(N)
    a[i] = 1.0
    return a

lref = numpy.array([one(len(i2l), l2i[l]) for l in labels])



import tensorflow as tf
sess = tf.InteractiveSession()



x_image = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
y_ = tf.placeholder(tf.float32, shape=[None, len(i2l)])

def bias_variable(shape):
  initial = tf.constant(0.0001, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev = 0.1))#2.0 / (25.0)
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev = 0.1))#2.0 / (25.0)
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev = 0.1))#2.0 / (25.0)
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev = 0.1))#2.0 / (25.0)
b_conv4 = bias_variable([256])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev = 0.1))#2.0 / (25.0)
b_conv5 = bias_variable([512])

h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

pool = tf.nn.max_pool(h_conv5, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

#W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 512], stddev = 0.01))#2.0 / (256.0)
#b_fc1 = bias_variable([512])

#h_pool4_flat = tf.reshape(h_pool4, [-1, 8 * 8 * 64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(tf.reshape(pool, [-1, 512]), keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([512, len(i2l)], stddev = 0.01))#2.0 / (256.0)
b_fc2 = bias_variable([len(i2l)])

y_log = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_conv = tf.nn.softmax(y_log)

yloss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_log, y_))
cross_entropy = yloss
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


indices = range(0, len(labels))
numpy.random.shuffle(indices)
#%%

sess.run(tf.initialize_all_variables())

batch_size = 200

yls = []

f = open('stdout', 'w')

for r in range(100):
    for i in range(0, len(labels), batch_size):
        xs = features[indices[i : i + batch_size]]
        ys = lref[indices[i : i + batch_size]]

        train, yl, acc = sess.run([train_step, yloss, accuracy], feed_dict = {x_image : xs, y_ : ys, keep_prob: 0.5})

        yls.append(yl)
        print yl, acc

        f.write('{0} {1} {2} {3}\n'.format(yl, acc, r, float(i) / len(labels)))

        if i % (10 * batch_size) == 0:
            plt.plot(yls)
            plt.title('epoch {0}.{1}'.format(r, float(i) / len(labels)))
            plt.show()
            print ''

f.close()

#%%
acc = sess.run([accuracy], feed_dict = {x_image: features[indices[500:1000]], y_: lref[indices[500:1000]], keep_prob: 1.0})
print acc
#%%
im = skimage.io.imread('screencast_frames/videoframe00095.bmp') / 255.0
#dy, dx, da = numpy.gradient(im)
#gradient = numpy.max(dx**2 + dy**2, axis = 2).reshape((im.shape[0], im.shape[1], 1))
#im = numpy.concatenate((im, gradient), axis = 2)

in_image = tf.placeholder(tf.float32, shape = [1, im.shape[0], im.shape[1], im.shape[2]])

h_conv1t = tf.nn.relu(conv2d(in_image, W_conv1) + b_conv1)
h_pool1t = max_pool_2x2(h_conv1t)

h_conv2t = tf.nn.relu(conv2d(h_pool1t, W_conv2) + b_conv2)
h_pool2t = max_pool_2x2(h_conv2t)

h_conv3t = tf.nn.relu(conv2d(h_pool2t, W_conv3) + b_conv3)
h_pool3t = max_pool_2x2(h_conv3t)

h_conv4t = tf.nn.relu(conv2d(h_pool3t, W_conv4) + b_conv4)
h_pool4t = max_pool_2x2(h_conv4t)

h_conv5t = tf.nn.relu(conv2d(h_pool4t, W_conv5) + b_conv5)

poolt = tf.nn.avg_pool(h_conv5t, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME')

h_fc1t = tf.reshape(poolt, [-1, 512])

y_logt = tf.matmul(h_fc1t, W_fc2) + b_fc2

out = tf.reshape(y_logt, [1, (im.shape[0] + 15) / 16, (im.shape[1] + 15) / 16, len(i2l)])


tmp = time.time()
res = out.eval(feed_dict = { in_image : [im] })[0, :, :, :]

res2 = numpy.argmax(res, axis = 2)
print time.time() - tmp

plt.imshow(im[:, :, :3])
plt.imshow(res2, interpolation = 'NONE', alpha = 0.5, extent = (60, im.shape[1] - 60, im.shape[0] - 60, 60))
plt.show()

#%%
idxs = numpy.where(numpy.array(labels) == i2l[73][1])[0]
for i in idxs:
    plt.imshow(features[i][:, :, :3])
    plt.show()
    1/0
#%%
vmin = numpy.min(res.flatten())
vmax = numpy.max(res.flatten())
for i in [857]:#182, 12, 58, 154, 141, 78, 82, 143, 93]:#range(features.shape[0]):#857 knight#413 noise
    idxs = numpy.where(numpy.array(labels) == i2l[i][1])[0]
    print i
    plt.imshow(im[:, :, :3])
    plt.show()
    plt.imshow(im[:, :, :3])
    plt.imshow(res[:, :, i], interpolation = 'NONE', alpha = 0.5, extent = (0, im.shape[1], im.shape[0], 0), vmin = -5.0, vmax = 2.5)
    plt.colorbar()
    plt.show()
    #for j in idxs:
    plt.imshow(features[idxs[15]][:, :, :3])
    plt.show()
#%%
saver = tf.train.Saver({ 'wc1' : W_conv1, 'bc1' : b_conv1, 'wc2' : W_conv2, 'bc2' : b_conv2, 'wc3' : W_conv3, 'bc3' : b_conv3, 'wc4' : W_conv4, 'bc4' : b_conv4, 'wc5' : W_conv5, 'bc5' : b_conv5, 'wfc1' : W_fc2, 'bfc1' : b_fc2 })
#%%
saver.restore(sess, 'tf.coeffs.100')
#%%
saver.save(sess, 'tf.coeffs.727')

f = open('tf.labels.727', 'w')
json.dump(classes, f)
f.close()
