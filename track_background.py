#%%
import imageio
import os
import numpy
import time
import json
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/lineage_process')

import features

vid = imageio.get_reader('saturday_april_23_2.mp4', 'ffmpeg')

frames = []
for im in vid:
    frames.append(im[:360])
    print len(frames)
    plt.imshow(frames[-1])
    plt.show()
#%%

import skimage.color
import scipy.signal

xs = numpy.linspace(-frames[0].shape[0] / 2.0, frames[0].shape[0] / 2.0, frames[0].shape[0]) + 50
ys = numpy.linspace(-frames[0].shape[1] / 2.0, frames[0].shape[1] / 2.0, frames[0].shape[1])

Ys, Xs = numpy.meshgrid(xs, ys, indexing = 'ij')

R = numpy.sqrt(Xs**2 + Ys**2)

G = numpy.exp(-R**2 / (2 * 50**2))

G /= numpy.linalg.norm(G.flatten())

plt.imshow(G)

#%%

import time

idxs = range(120, 150, 2)

hy = frames[0].shape[0] / 2.0
hx = frames[0].shape[1] / 2.0

dxs = []

for i1, i0 in zip(idxs[1:], idxs[:-1]):
    tmp = time.time()
    f1 = frames[i1]#skimage.color.rgb2hsv(frames[i1])
    f0 = frames[i0]#skimage.color.rgb2hsv(frames[i0])

    out = scipy.signal.fftconvolve(f1[:, :, 0] * G, f0[:, :, 0] * G, mode = 'same')

    c = numpy.unravel_index(numpy.argmax(out), out.shape)

    dxs.append([c - numpy.array([hy, hx])])

    #print time.time() - tmp
    print dxs[-1]

    #plt.subplot(2, 1, 1)
    #plt.imshow(f1)
    #plt.subplot(2, 1, 2)
    #plt.imshow(f0)
    #plt.show()
    #plt.imshow(out)
    #plt.colorbar()
    #plt.show()
