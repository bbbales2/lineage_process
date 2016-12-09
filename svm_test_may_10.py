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

import features
#%%

backgrounds = []

for f in os.listdir('screencast_frames'):
    backgrounds.append(skimage.io.imread('screencast_frames/{0}'.format(f))[40:360, :300])

#%%
codes = os.listdir('filtered_sprites_160x160')

Xs = []
Xsb = []
labels = []
for code in codes:
    images = numpy.load('filtered_sprites_160x160/{0}'.format(code)).astype('double')[:, 40:120, 40:120, :4]
    
    labels.extend(images.shape[0] * [int(code)])
    
    images2 = []
    
    for image in images:
        images2.append(features.rgbhist(image[:, :, :3] / 255.0, 4))
    
    Xs.append(images2)
    
    backgroundSample = backgrounds[1][20:100, 20:100]
    
    images3 = []
    for image in images:
        mask = (image[:, :, 3] > 254.5)
        image2 = numpy.array(image).copy()
        image2[:, :, 0] += mask * backgroundSample[:, :, 0].astype('double')
        image2[:, :, 1] += mask * backgroundSample[:, :, 1].astype('double')
        image2[:, :, 2] += mask * backgroundSample[:, :, 2].astype('double')
        images3.append(features.rgbhist(image2[:, :, :3] / 255.0, 4))

    Xsb.append(images3)

    plt.imshow(images3[0], interpolation = 'NONE')
    plt.show()
    print ''
    
Xs = numpy.concatenate(Xs)
Xsb = numpy.concatenate(Xsb)

#%%

Xs2 = numpy.reshape(Xs, (Xs.shape[0], -1))
Xs3 = numpy.reshape(Xsb, (Xsb.shape[0], -1))

#%%
svm.predict(Xs3)
import sklearn.svm
import sklearn.cross_validation

svm = sklearn.svm.LinearSVC(C = 1.0)

print sklearn.cross_validation.cross_val_score(svm, Xs2, labels)

#%%

svm.fit(Xs2, labels)

#%%

print 1.0 - numpy.count_nonzero(svm.predict(Xs3) - labels) / float(len(labels))