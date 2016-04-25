#%%
import os
import re
import numpy
import matplotlib
import json
import itertools
import time
import skimage.io
import sklearn.mixture
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.svm
import sklearn.lda

os.chdir('/home/bbales2/lineage_process')

import features
os.chdir('/home/bbales2/backup/lineage_process')

#%%

f = open('reference.json')
reference = json.load(f)
f.close()

classes = []

for ref in reference:
    if reference[ref] == 'true':
        classes.append(ref.split('/')[1])

numpy.random.shuffle(classes)

classes = classes[:50]

classes = sorted(classes)

#%%
backgrounds = []

for f in os.listdir('screencast_frames'):
    backgrounds.append(skimage.io.imread('screencast_frames/{0}'.format(f))[40:360, :300])
#%%

dframes = {}

labels = []
numImages = 0
rpts = 1
for i, code in enumerate(classes):
    f = open('all_sprites_160x160_index/{0}.index'.format(code))
    data = json.load(f)
    f.close()

    images = numpy.load('filtered_sprites_160x160/{0}'.format(code))

    numImages += rpts * images.shape[0]

    labels.extend([code] * rpts * images.shape[0])

    print i

eimages = []

idx = 0
for i, code in enumerate(classes):
    f = open('all_sprites_160x160_index/{0}.index'.format(code))
    data = json.load(f)
    f.close()

    images = numpy.load('filtered_sprites_160x160/{0}'.format(code))

    fs = []

    for im in images:
        im2 = im[80 - 32 : 80 + 32, 80 - 32 : 80 + 32, :4]
        mask = (im2[:, :, 3] > 254.5)
        for r in range(rpts):
            background = backgrounds[numpy.random.randint(0, len(backgrounds))]

            ii = numpy.random.randint(0, background.shape[0] - 64)
            jj = numpy.random.randint(0, background.shape[1] - 64)

            backgroundSample = background[ii : ii + 64, jj : jj + 64]

            im3 = numpy.array(im2)
            #im3[:, :, 3] = ((1.0 - mask) * 255.0).astype('uint8')
            im3[:, :, 0] += mask.astype('uint8') * backgroundSample[:, :, 0]
            im3[:, :, 1] += mask.astype('uint8') * backgroundSample[:, :, 1]
            im3[:, :, 2] += mask.astype('uint8') * backgroundSample[:, :, 2]
            #plt.imshow(mask)
            #plt.colorbar()
            #plt.show()
            #plt.imshow(im3)
            #plt.show()
            #plt.imshow(backgroundSample)
            #plt.show()
            im4 = im3[:, :, :3] / 255.0
            eimages.append(im4)
            idx += 1
            #1/0

    print i

images = numpy.array(eimages)
#%%
images = []
for image in eimages:
    images.append(features.rgbhist(image, 8))
images = numpy.array(images)
#%%
del eimages
#%%

#features = numpy.array([(feature - mean) for feature in features])
i2l = list(enumerate(list(set(labels))))
l2i = dict([(l, i) for i, l in i2l])

def one(N, i):
    a = numpy.zeros(N)
    a[i] = 1.0
    return a

lref = numpy.array([one(len(i2l), l2i[l]) for l in labels])

#%%
colors = []
for k in range(len(images)):#numpy.random.choice(range(len(images)), 1000):
    image = images[k]
    for i, j in zip(*numpy.where(numpy.sum(image, axis = 2) != 0)):
        colors.append(image[i, j])
    #plt.imshow(image)
    #plt.show()

plt.imshow(images[1])

N = 25
bgkmeans = sklearn.cluster.MiniBatchKMeans(N)
bgkmeans.fit(colors)
plt.imshow(numpy.abs(bgkmeans.cluster_centers_.reshape((1, N, 3))), interpolation = 'NONE')
plt.show()
#%%
chists = []
for k in range(len(images)):#numpy.random.choice(range(len(images)), 1000):
    image = images[k]
    #ccolors = []
    #for i, j in zip(*numpy.where(numpy.sum(image, axis = 2) != 0)):
    #    ccolors.append(image[i, j])

    #f = numpy.zeros(N)
    #if len(ccolors) != 0:
    #    for l in bgkmeans.predict(ccolors):
    #        f[l] += 1.0
    #    f /= len(ccolors)

    chists.append(get_chist(image))
#%%
#svm = sklearn.svm.SVC()
#print sklearn.cross_validation.cross_val_score(svm, chists, labels)

lr = sklearn.linear_model.LogisticRegression()
print sklearn.cross_validation.cross_val_score(lr, chists, labels)
#%%
lr.fit(chists, [l2i[l] for l in labels])
#%%

frames = numpy.load('/home/bbales2/backup/lineage_process/april_23_recorded.npy')
#%%
b = 8
for frame in frames[0:100:10]:
    im = frame[:360].astype('double') / 255.0

    tmp = time.time()
    hog = features.hogpad(features.hog(im, b))
    rgb = features.rgbhist(im, 8)
    print time.time() - tmp

    inp = rgb.reshape((rgb.shape[0] * rgb.shape[1], 3))
    #inp /= 1e-5 + numpy.linalg.norm(inp, axis = 1).reshape((inp.shape[0], 1))
    labels = numpy.array(bgkmeans.predict(inp))
    labelsplot = numpy.abs(numpy.array([numpy.abs(bgkmeans.cluster_centers_[l]) for l in labels]).reshape(rgb.shape))
    labels = labels.reshape((rgb.shape[0], rgb.shape[1]))

    plt.imshow(labelsplot, interpolation = 'NONE', extent = (0, im.shape[1], im.shape[0], 0))
    plt.show()
    plt.imshow(im)
    #plt.imshow(rgb, interpolation = 'NONE')
    plt.imshow(labelsplot, interpolation = 'NONE', extent = (0, im.shape[1], im.shape[0], 0), alpha = 0.5)
    plt.show()

    chists = []
    R = 3
    for i in range(R, labels.shape[0] - R):
        for j in range(R, labels.shape[1] - R):
            chists.append(get_chist(rgb[i - R : i + R, j - R : j + R]))

    clabels = numpy.pad(lr.predict(chists).reshape(labels.shape[0] - 2 * R, labels.shape[1] - 2 * R), R, mode = 'edge')
    plt.imshow(im)
    plt.imshow(clabels, interpolation = 'NONE', extent = (0, im.shape[1], im.shape[0], 0), alpha = 0.5)
    plt.show()
    #1/0
#%%

chists = []
def get_chist(image):
    ccolors = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    chist = numpy.zeros(N)
    if len(ccolors) != 0:
        for l in bgkmeans.predict(ccolors):
            chist[l] += 1.0
        chist /= len(ccolors)

    return chist