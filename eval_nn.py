#%%
import os
import skimage.io
import skimage.transform
import numpy
import matplotlib.pyplot as plt
import sys
import json
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm

#%%

sys.path.append('/home/bbales2/caffe-tensorflow')

os.chdir('/home/bbales2/lineage_process')


import googlenet

import tensorflow as tf
sess = tf.InteractiveSession()

tens = tf.placeholder(tf.float32, shape=[100, 160, 160, 3])

# Create an instance, passing in the input data
with tf.variable_scope("image_filters", reuse = False):
    net = googlenet.GoogleNet({'data' : tens})

with tf.variable_scope("image_filters", reuse = True):
    net.load('googlenet.tf', sess, ignore_missing = True)

#%%
f = open('totrain.json')
files, folders, labels = json.load(f)
f.close()

images = numpy.load('toTrain.dat')

labels = labels
images = images
#%%

images2 = []
for im in images:
    im = im[:, :, ::-1].astype('float32')

    mean = numpy.array([104., 117., 124.])

    for c in range(3):
        im[:, :, c] -= mean[c]

    images2.append(numpy.array(im))
#%%
import time
tmp = time.time()
outputs = []
for i in range(0, len(images2), 100):
    iN = numpy.zeros((100, 160, 160, 3))

    iN[0 : min(100, len(images2) - i)] = images2[i : i + 100]

    output = sess.run(net.get_output(), feed_dict = { tens : iN })

    outputs.append(output.reshape(100, 1024))

    print i, len(images2)
print "took: ", time.time() - tmp

#%%

toClassify = numpy.vstack(outputs)[:len(images2)]
#sklearn.linear_model.
lr = sklearn.svm.LinearSVC()

Xtrain, Xtest, Ytrain, Ytest = sklearn.cross_validation.train_test_split(toClassify, labels)

lr.fit(Xtrain, Ytrain)


#%%

Yp = lr.predict(Xtest)
print sum(Yp != Ytest) / float(len(Ytest))


#%%

import sklearn.cross_validation

sklearn.cross_validation.cross_val_score(lr, toClassify, labels)

#%%

