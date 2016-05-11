#%%

import matplotlib.pyplot as plt
import os
import re
import numpy
import json
import itertools
import skimage.io
import scipy.ndimage
import skimage.exposure
import time
import pickle
import sklearn.cluster
import sklearn.mixture
import skimage.color
import itertools

os.chdir('/home/bbales2/lineage_process')

import features

f = open('/home/bbales2/backup/lineage_process/april_23_recorded.pkl')
presses = pickle.load(f)
f.close()

frames = numpy.load('/home/bbales2/backup/lineage_process/april_23_recorded.npy')
#%%

for (j, i), screen in zip(presses, frames):

    plt.imshow(screen)

    circle = plt.Circle((j, i), 10, color = 'r', fill = False)

    plt.gca().add_artist(circle)

    print i, j
    plt.show()

#%%

b = 8

sy = 7
sx = 3

indices = {}
descs = {}
backgrounds = []
for k, frame in enumerate(frames[0:100:10]):
    im = frame[:360].astype('double') / 255.0
    #im = skimage.color.rgb2hsv((im * 255).astype('uint8'))
    #plt.imshow(im[:, :, 0], interpolation = 'NONE', cmap = plt.cm.hsv);
    #plt.colorbar()
    #plt.show()
    #plt.imshow(im[:, :, 1], interpolation = 'NONE', cmap = plt.cm.hsv);
    #plt.colorbar()
    #plt.show()
    #plt.imshow(im[:, :, 2], interpolation = 'NONE', cmap = plt.cm.hsv);
    #plt.colorbar()
    #plt.show()

    tmp = time.time()
    hog = features.hogpad(features.hog(im, b))
    rgb = features.rgbhist(im, b)
    print time.time() - tmp

    #plt.imshow(im, interpolation = 'NONE')
    #plt.show()
    #plt.imshow(skimage.color.rgb2gray(im), interpolation = 'NONE', cmap = plt.cm.gray)
    #plt.imshow(rgb, interpolation = 'NONE')
    #plt.show()

    Y = rgb.shape[0]
    X = rgb.shape[1]

    rgbf = numpy.zeros((rgb.shape[0], rgb.shape[1], sy * sx * rgb.shape[2]))

    indices[k] = []
    descs[k] = []

    backgrounds.extend(rgb.reshape((rgb.shape[0] * rgb.shape[1], 3)))
    for i in range(sy / 2, rgb.shape[0] - sy / 2):
        for j in range(sx / 2, rgb.shape[1] - sx / 2):
            indices[k].append((i, j))
            descs[k].append(rgb[i - sy / 2 : i + sy / 2, j - sx / 2 : j + sx / 2].flatten())
            #descs[k].append(hog[i - sy / 2 : i + sy / 2, j - sx / 2 : j + sx / 2].flatten())
#sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
#%%
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
class KMedians(sklearn.cluster.KMeans):
    def _e_step(self, X):
        self.labels_ = manhattan_distances(X, self.cluster_centers_).argmin(axis=1)

    def _average(self, X):
        return np.median(X, axis=0)
#%%
backgrounds = numpy.array(backgrounds)
bg2 = numpy.concatenate((numpy.linalg.norm(backgrounds, axis = 1).reshape((backgrounds.shape[0], 1)), backgrounds), axis = 1)
bg2[:, 1] = bg2[:, 1] / (1e-5 + bg2[:, 0])
bg2[:, 2] = bg2[:, 2] / (1e-5 + bg2[:, 0])
bg2[:, 3] = bg2[:, 3] / (1e-5 + bg2[:, 0])
#%%
plt.hist(backgrounds[:, 2])
plt.show()
#%%
N = 15
bgkmeans = sklearn.cluster.KMeans(N)
bgkmeans.fit(bg2[:, 1:])
plt.imshow(numpy.abs(bgkmeans.cluster_centers_.reshape((1, N, 3))), interpolation = 'NONE')
plt.show()

#%%
for frame in frames[0:100:10]:
    im = frame[:360].astype('double') / 255.0

    tmp = time.time()
    hog = features.hogpad(features.hog(im, b))
    rgb = features.rgbhist(im, b)
    print time.time() - tmp

    inp = rgb.reshape((rgb.shape[0] * rgb.shape[1], 3))
    inp /= 1e-5 + numpy.linalg.norm(inp, axis = 1).reshape((inp.shape[0], 1))
    labels = numpy.array(bgkmeans.predict(inp))
    labelsplot = numpy.abs(numpy.array([numpy.abs(bgkmeans.cluster_centers_[l]) for l in labels]).reshape(rgb.shape))

    plt.imshow(labelsplot, interpolation = 'NONE', extent = (0, im.shape[1], im.shape[0], 0))
    plt.show()
    plt.imshow(im)
    plt.imshow(labelsplot, interpolation = 'NONE', extent = (0, im.shape[1], im.shape[0], 0), alpha = 0.5)
    plt.show()
#%%
s = sklearn.cluster.SpectralClustering(5)
s.fit(backgrounds)
plt.imshow(s.cluster_centers_.reshape((1, 5, 3)), interpolation = 'NONE')
plt.show()
#%%
gmm = sklearn.mixture.GMM(5, covariance_type = 'full')

gmm.fit(bg2[:, 1:])
plt.imshow(gmm.means_.reshape((1, gmm.means_.shape[0], gmm.means_.shape[1])), interpolation = 'NONE')
plt.show()
#%%
out = list(itertools.chain(*descs.values()[0:10]))
numpy.random.shuffle(out)
out = out[0:10000]
#pca = sklearn.decomposition.PCA(n_components = 10, copy = True, whiten = True)
#pca.fit(out)

#nout = pca.transform(out)
kmeans = sklearn.cluster.KMeans(3)
kmeans.fit(nout)

plt.plot(nout[:, 1], nout[:, 0], '.')
plt.show()
#%%
for k, frame in enumerate(frames[0:100:10]):
    im = frame[:360].astype('double') / 255.0
    tmp = time.time()
    hog = features.hogpad(features.hog(im, b))
    rgb = features.rgbhist(im, b)
    print time.time() - tmp

    plt.imshow(im, interpolation = 'NONE')
    plt.show()
    plt.imshow(skimage.color.rgb2gray(im), interpolation = 'NONE', cmap = plt.cm.gray)
    #plt.imshow(rgb, interpolation = 'NONE')
    #plt.show()

    inds = indices[k]
    desc = descs[k]

    labels = kmeans.predict(pca.transform(desc))

    labels = labels.reshape(rgb.shape[0] - sy + 1, rgb.shape[1] - sx + 1)

    labels = numpy.pad(labels, [[sy / 2, sy / 2], [sx / 2, sx / 2]], mode = 'edge')

    plt.imshow(labels, alpha = 0.35, interpolation = 'NONE', extent = (0, im.shape[1], im.shape[0], 0))
    plt.show()

    #1 / 0
    #for