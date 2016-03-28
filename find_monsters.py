#%%

b = 8
bins = 8


num_detectors = nwd * nhd

num_classes = 200

dimension = dwb * dhb * bins

print num_detectors * num_classes * dimension

#%%

import os

os.chdir('/home/bbales2/lineage_process/')

import sklearn.svm
import matplotlib.pyplot as plt
import numpy
import skimage.io
import skimage.feature
import skimage.color
import time
import mahotas
#%%

im = skimage.io.imread('screencast_frames/videoframe00049.bmp')
im = im[:(img.shape[0] / b) * b, :(img.shape[1] / b) * b]
img = skimage.color.rgb2gray(im)

#%%
def build_descriptors(im, b):
    w = im.shape[1]
    h = im.shape[0]

    wb = w / b
    hb = h / b

    dw = b * 10
    dh = b * 10

    dwb = dw / b
    dhb = dh / b

    nwd = wb - dwb + 1
    nhd = hb - dhb + 1

    img = skimage.color.rgb2gray(im)
    hogs = skimage.feature.hog(img, orientations = bins, pixels_per_cell = (b, b), cells_per_block = (1, 1), feature_vector = False)

    descriptors = []
    for i in range(nhd):
        for j in range(nwd):
        #window = im[i * b : i * b + dh, j * b : j * b + dw]
            descriptors.append(hogs[i : i + dhb, j : j + dwb].flatten())

        #skimage.io.imsave('tmp/{0}.{1}.png'.format(i, j), window)

    return descriptors, (nhd, nwd), (dh, dw)#hogs = hogs.reshape((hogs.shape[0], hogs.shape[1], hogs.shape[4]))

labels = numpy.zeros((nhd, nwd)).astype('int')

tmp = time.time()
descriptors, (nhd, nwd), (dh, dw) = build_descriptors(im, b)
print 'descriptor time', time.time() - tmp

labels[16:22, 30:38] = 1

labels2 = labels.flatten()
#%%


tmp = time.time()
descriptors, (nhd, nwd), (dh, dw) = build_descriptors(im, b)
print 'descriptor time', time.time() - tmp

#%%
#desc = []
#labels = []

for i in range(1, 100, 1):
    im = skimage.io.imread('screencast_frames/videoframe{0:0>5d}.bmp'.format(i))
    im = im[:(img.shape[0] / b) * b, :(img.shape[1] / b) * b]
    img = skimage.color.rgb2gray(im)

    descriptors, (nhd, nwd), (dh, dw) = build_descriptors(im, b)

    for ii in range(17, 22):
        for jj in range(34, 36):
            desc.append(descriptors[ii * nwd + jj])
            labels.append(1)

    #for o in range(50):
    #    while 1:
    #        ii = numpy.random.randint(nhd)
    #        jj = numpy.random.randint(nwd)

    #        if numpy.sqrt((ii - 19)**2 + (jj - 34)**2) > 5.0:
    #            break

    #    desc.append(descriptors[ii * nwd + jj])
    #    labels.append(0)


#%%
desc = []
labels = []

for i in range(0, 3, 1):
    im = skimage.io.imread('labeled/{0}.bmp'.format(i))
    im = im[:(im.shape[0] / b) * b, :(im.shape[1] / b) * b]

    plt.imshow(im)

    lg = skimage.io.imread('labeled/{0}_map.png'.format(i))
    lg = lg[:(im.shape[0] / b) * b, :(im.shape[1] / b) * b]

    lg = lg[:, :, 3]

    lg = mahotas.distance(lg) > 100

    descriptors, (nhd, nwd), (dh, dw) = build_descriptors(im, b)

    fig = plt.gcf()
    for ii in range(nhd):
        for jj in range(nwd):
            desc.append(descriptors[ii * nwd + jj])
            labels.append(lg[ii * b + dh / 2 + b / 2, jj * b + dw / 2 + b / 2])
            if labels[-1]:
                circle = plt.Circle((jj * b + dw / 2 + b / 2, ii * b + dh / 2 + b / 2), 1.0, color='r')
                fig.gca().add_artist(circle)

    plt.show()

svm = sklearn.svm.LinearSVC(class_weight = 'balanced')

print sklearn.cross_validation.cross_val_score(svm, desc, labels)

svm.fit(desc, labels)

#%%
plt.imshow(mahotas.distance(lg) > 80)
#%%
plt.imshow(im)
plt.imshow(mahotas.erode(mahotas.erode(mahotas.erode(mahotas.erode(mahotas.erode(mahotas.erode(lg[:, :])))))), alpha = 0.5)

#%%
for i in range(40, 60, 1):
    #im = skimage.io.imread('labeled/{0}.bmp'.format(i))
    im = skimage.io.imread('screencast_frames/videoframe{0:0>5d}.bmp'.format(i))
    im = im[:(im.shape[0] / b) * b, :(im.shape[1] / b) * b]
    tmp = time.time()
    descriptors, (nhd, nwd), (dh, dw) = build_descriptors(im, b)

    nlabels = svm.predict(descriptors)

    nlabels = nlabels.reshape((nhd, nwd))
    print 'time', time.time() - tmp
    #nlabels = numpy.tile(nlabels, [1, 1, b, b])
    #nlabels = nlabels.reshape(im.shape[0:2])

    toplot = skimage.transform.resize(nlabels.astype('float'), im.shape[0:2])

    plt.imshow(im)
    fig = plt.gcf()
    plt.imshow(nlabels, interpolation = 'NONE', alpha = 0.5, extent = (dw / 2, nwd * b + dw / 2, nhd * b + dh / 2, dh / 2))
    #for ii in range(nhd):
    #    for jj in range(nwd):
            #desc.append(descriptors[ii * nwd + jj])
            #labels.append(lg[ii * b + dh / 2 + b / 2, jj * b + dw / 2 + b / 2])
    #        if nlabels[ii, jj]:
    #            circle = plt.Circle((jj * b + dw / 2 + b / 2, ii * b + dh / 2 + b / 2), 1.0, color='r')
    #            fig.gca().add_artist(circle)
    plt.show()
    print 'hi'

#%%
