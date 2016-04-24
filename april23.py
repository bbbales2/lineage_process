#%%
import imageio
import os
import numpy
import time
import json

os.chdir('/home/bbales2/lineage_process')

import features

vid = imageio.get_reader('saturday_april_23_2.mp4', 'ffmpeg')

frames = []
for im in vid:
    frames.append(im)
#im = vid.get_data(0)

#%%
dxs = []
for i in range(1000):
    dxs.append(numpy.linalg.norm(frames[i + 1] - frames[i], axis = 2).flatten().mean())

plt.plot(dxs)
plt.show()

start = numpy.where(dxs > numpy.mean(dxs))[0][0]

#%%
fps = vid.get_meta_data()['fps']
#%%
import re

f = open('saturday_april_23_2.txt')
lines = f.read()
f.close()

x = 0
y = 0

clicks = []

for line in lines.split('\n'):
    if line[0:5] == 'mouse':
        out = re.search('REL_([XY]) ([\-0-9]+)', line)

        if out:
            t, d = out.groups()

            if t == 'X':
                x += int(d)
            else:
                y += int(d)

        click = re.search('([\.0-9]+) [0-9] EV_KEY 272 .* 1', line)

        if click:
            t = click.groups()[0]

            clicks.append((float(t), (y, x)))

print x, y

clicks = sorted(clicks, key = lambda x : x[0])

clicks = [(int(start + (t - clicks[0][0]) * fps), ij) for t, ij in clicks]

#%%
import skimage.filters

rDx = 0.0

rDxs = []

for t in xrange(len(frames) - 1):
    f1 = skimage.filters.gaussian(frames[t + 1][400:480, 140:480, :], 1.0)
    f0 = skimage.filters.gaussian(frames[t][400:480, 140:480, :], 1.0)
    d = numpy.linalg.norm(f1 - f0, axis = 2)
    rdx = numpy.mean(d.flatten())

    #rDx = rDx * 0.9 + rdx * 0.1

    rDxs.append(rdx)

    #print t

    if rdx > 0.02:
        print t
        plt.imshow(d)
        plt.show()
        plt.imshow(f1, interpolation = 'NONE')
        plt.show()
        plt.imshow(f0, interpolation = 'NONE')
        plt.show()

#%%
clickrs = []
for i in range(len(clicks) - 1):
    t, (ix, jx) = clicks[i]
    tn = clicks[i + 1][0]
    clickrs.append((t, (ix, jx), sum(rDxs[t : tn])))

#%%
for t, (i, j), v in clickrs:
    plt.imshow(frames[t])
    if v > numpy.mean(rDxs):
        circle = plt.Circle((j, i), 20, color = 'r', fill = False)
    else:
        circle = plt.Circle((j, i), 20, color = 'b', fill = False)

    plt.gca().add_artist(circle)

    print i, j
    plt.show()

#%%
ts, vs = zip(*[(t, 0.01) for (t, ij) in clicks])
plt.plot(rDxs)
plt.plot(ts, vs, 'o')
plt.show()
#%%

b = 8

rDx = 0.0

rDxs = []

sy = 7
sx = 3

for t in xrange(10000):
    #im = images[t, :360, :, :].astype('double')
    #tmp = time.time()
    #hog = features.hogpad(features.hog(im, b))
    #rgb = features.rgbhist(im, b)
    #print time.time() - tmp

    Y = rgb.shape[0]
    X = rgb.shape[1]

    rgbf = numpy.zeros((rgb.shape[0], rgb.shape[1], sy * sx * rgb.shape[2]))
    for i in range(rgb.shape[0]