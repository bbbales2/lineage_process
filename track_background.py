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

xs = numpy.linspace(-frames[0].shape[0] / 2.0, frames[0].shape[0] / 2.0, frames[0].shape[0])
ys = numpy.linspace(-frames[0].shape[1] / 2.0, frames[0].shape[1] / 2.0, frames[0].shape[1])

Ys, Xs = numpy.meshgrid(xs, ys, indexing = 'ij')

R = numpy.sqrt(Xs**2 + Ys**2)

G = numpy.exp(-R**2 / (2 * 50**2))

G /= numpy.linalg.norm(G.flatten())

plt.imshow(G)

#%%
import matplotlib.animation as animation
import skimage.transform

import time
import skimage.io
reload(bgalign)

idxs = range(120, 250, 1)

hy = frames[0].shape[0] / 2.0
hx = frames[0].shape[1] / 2.0

dxs = []

plt.imshow(f1)
fig = plt.gcf()

try:
    writer = imageio.get_writer('movie.mp4', fps = 5.0)

    cmap = plt.cm.jet

    for i1, i0 in zip(idxs[1:], idxs[:-1]):
        tmp = time.time()
        f1 = frames[i1]#skimage.color.rgb2hsv(frames[i1])
        f0 = frames[i0]#skimage.color.rgb2hsv(frames[i0])

        res, o = bgalign.offset(24, f1, f0)
        intensities = label.image(8, f1, refs)

        #plt.imshow(f1, interpolation = 'NONE')
        #plt.imshow(, interpolation = 'NONE', vmin = 0, vmax = refs.shape[1], extent = (0, f1.shape[1], f1.shape[0], 0), alpha = 0.5)
        #labels = numpy.argmax(intensities, axis = 2) / float(refs.shape[1])
        labels = intensities[:, :, 2] / intensities[:, :, 2].max()
        labels = skimage.transform.resize(labels, (f1.shape[0], f1.shape[1]), order = 0)

        towrite = f1 * 0.5 / 255.0 + cmap(labels)[:, :, :3] * 0.5
        #plt.imshow(towrite)
        #plt.show()
        writer.append_data(towrite)
        print i0
finally:
    writer.close()
#%%
    #plt.show()

with writer.saving(fig, "writer_test.mp4", 10):
    for i in range(10):
        fig.set_data(frames[i])
        writer.grab_frame()
    #skimage.io.imsave('/home/bbales2/lineage_process/dec9/{0}.png'.format(i1), f1)

    #plt.subplot(2, 1, 1)
    #plt.imshow(f1)
    #plt.plot(pts[:, 1], pts[:, 0], 'r+')
    #plt.subplot(2, 1, 2)
    #plt.imshow(f0)
    #plt.gcf().set_size_inches((15, 10))
    #plt.show()
    #plt.imshow(res, interpolation = 'NONE')
    #plt.colorbar()
    #plt.show()

    #print o

    #out = scipy.signal.fftconvolve(f1[:, :, 0] * G, f0[:, :, 0], mode = 'same')

    #c = numpy.unravel_index(numpy.argmax(out), out.shape)

    #dxs.append([c - numpy.array([hy, hx])])

    #print time.time() - tmp
    #print dxs[-1], i1, i0

    #plt.imshow(out)
    #plt.colorbar()
    #plt.show()

#%%

import label
#%%
reload(label)

refs = []

def buildHist(im):
    hist = numpy.zeros((16 * 16 * 16))

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im.shape[2] == 4 and im[i, j, 3] < 5:
                continue

            c = im[i, j] / 16

            hist[c[2] * 16 * 16 + c[1] * 16 + c[0]] += 1

    return hist / numpy.linalg.norm(hist)

fnames = ['grass', 'dirt', 'kobold', 'orc', 'gnoll', 'knight']

for fname in fnames:
    sample = skimage.io.imread('dec9/samples/{0}/0.png'.format(fname))

    refs.append(label.buildHist(sample))

#dirt = skimage.io.imread('dec9/samples/dirt/0.png')

#ghist = buildHist(grass)
#dhist = buildHist(dirt)

refs = numpy.array(refs).T
#%%

reload(label)
b = 8

f1 = frames[300]

M = f1.shape[0]
N = f1.shape[1]

mb = M / b
nb = N / b

hists = numpy.zeros((mb, nb, refs.shape[0]))

tmp = time.time()
for bi in range(mb):
    for bj in range(nb):
        hists[bi, bj] = label.buildHist(f1[bi * b : (bi + 1) * b, bj * b : (bj + 1) * b])
print time.time() - tmp

tmp = time.time()
labels = label.image(8, f1, refs)
print time.time() - tmp

print bi

#%%
labels = numpy.zeros((mb, nb, len(refs)))
for bi in range(mb):
    for bj in range(nb):
        for i in range(len(refs)):
            labels[bi, bj, i] = hists[bi, bj].dot(refs[i])
#%%
tmp = time.time()
labels = label.image(8, f1, refs)
print time.time() - tmp
#%%
plt.imshow(f1, interpolation = 'NONE')
plt.imshow(numpy.argmax(labels, axis = 2), interpolation = 'NONE', extent = (0, f1.shape[1], f1.shape[0], 0), alpha = 0.5)
plt.show()

#%%
for i, fname in enumerate(fnames):
    plt.imshow(f1, interpolation = 'NONE')
    plt.imshow(labels[:, :, i], interpolation = 'NONE', extent = (0, f1.shape[1], f1.shape[0], 0), alpha = 0.5)
    plt.title(fname)
    plt.show()
#%%

pts = []

M = frames[0].shape[0]
N = frames[0].shape[1]
R = 12

for i in range(R, M - R, 48):
    for j in range(R, N - R, 48):
        pts.append((i, j))

pts = numpy.array(pts)

result = numpy.zeros((2 * R, 2 * R))
for dx in range(-R, R):
    for dy in range(-R, R):
        loss = 0.0
        #f0s = []
        #f1s = []
        for pt in pts:
            loss += numpy.linalg.norm(f0[pt[0], pt[1]] - f1[pt[0] + dy, pt[1] + dx])
            #f0s.append(f0[pt[0], pt[1]])
            #f1s.append(f1[pt[0] + dy, pt[1] + dx])

        result[dy + R, dx + R] = loss

        #f1[pts + numpy.array([dx, dy])]

    print dx

#%%

print numpy.unravel_index(numpy.argmin(result), result.shape) - numpy.array([R, R])

plt.imshow(result, interpolation = 'NONE')
plt.colorbar()
plt.show()

#%%

import pyximport
pyximport.install(reload_support = True)#

import bgalign

#%%

reload(bgalign)

tmp = time.time()
res = bgalign.loffset(12, f0, f1)
print time.time() - tmp

plt.imshow(res[1], interpolation = 'NONE')

print res

#print numpy.unravel_index(numpy.argmin(res[0]), res[0].shape) - numpy.array([R, R])

#plt.imshow(res[0], interpolation = 'NONE')
#plt.colorbar()
#plt.show()

