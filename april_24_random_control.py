#%%

import imageio
import os
import numpy
import time
import json
import bisect
import skimage.filters
import skimage.color

os.chdir('/home/bbales2/lineage_process')

import features

os.chdir('/home/bbales2/backup/lineage_process')

vid = imageio.get_reader('lineage_rec_1.mp4', 'ffmpeg')

im = vid.get_data(0)

#%%

numImages = 15318

#%%

images = numpy.memmap('frames.npy', dtype = 'float32', mode = 'r', shape = (numImages, im.shape[0], im.shape[1], im.shape[2]))

#%%
images2 = []
for image in images[1040:1200:1]:
    images2.append(image[:360].astype('double'))
    #plt.imshow(images2[-1])
    #plt.show()
    #1/0

images2 = numpy.array(images2)

dx1 = images2[1:] - images2[0:-1]

dx2 = []
for dx in dx1:
    dx2.append(numpy.linalg.norm(features.rgbhist(dx, 16), axis = 2))

im = dx2[0]
ims = []
#%%
Xs, Ys = numpy.meshgrid(numpy.arange(im.shape[1]), numpy.arange(im.shape[0]))
Rs = (Xs - 18.5)**2 / 2.0 + (Ys - 11.5)**2
Rs2 = (Xs - 18.5)**2 / 1.5 + (Ys - 11.5)**2 / 2.0
inf = numpy.exp(-Rs / 100.0) - numpy.exp(-Rs2 / 10.0)
#plt.imshow(inf)

#idx = numpy.array([11.5, 18.5])

for image, im2 in zip(images2, dx2):
    im = im * 0.75 + im2 * 0.25
    #m = numpy.linalg.norm(im, axis = 2)
    im2 = numpy.array(im)
    im2[10:15, 19:20] = 0.0
    probs = (inf * im2).flatten()

    if numpy.random.rand() < 0.1:
        csum = numpy.cumsum(probs)

        choice = numpy.random.rand() * csum[-1]

        idx2 = numpy.unravel_index(bisect.bisect_left(csum, choice), im2.shape)
    else:
        idx2 = numpy.unravel_index(numpy.argmax(probs), im2.shape)

    idx = numpy.array(idx2)#0.75 * idx + 0.25 *

    im2[int(idx[0]), int(idx[1])] = 100.0

    plt.imshow(image)
    plt.imshow(inf * im2, interpolation = 'NONE', alpha = 0.5, extent = (0, image.shape[1], image.shape[0], 0))
    plt.show()
    print 'hi'
    #ims.append(im)
#%%