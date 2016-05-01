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
    frames.append(im[:360])
    plt.imshow(frames[0])
    plt.show()
#%%
import sklearn.cluster

im = frames[0]

kmeans = sklearn.cluster.KMeans(64)
data = im.reshape((im.shape[0] * im.shape[1], 3))
kmeans.fit(data)
#%%
plt.imshow(kmeans.cluster_centers_.reshape((1, 64, 3)), interpolation = 'NONE')
#%%
im2 = numpy.reshape([kmeans.cluster_centers_[i] for i in kmeans.predict(data)], im.shape).astype('uint8')
plt.imshow(im2)
plt.show()
#%%
rgb = features.rgbhist(im2.astype('double'), 2).astype('uint8')

labels2 = numpy.reshape(kmeans.predict(rgb.reshape((rgb.shape[0] * rgb.shape[1], 3))), rgb.shape[:2])

plt.imshow(rgb.astype('uint8'), interpolation = 'NONE')
plt.show()
#%%

#%%
plt.imshow(im)
plt.show()
#im = vid.get_data(0)
#%%
def buildDochists(dlabels, nLabels, w = 20, h = 20):
    labels = set(dlabels.flatten())

    data = numpy.zeros((dlabels.shape[0] - h, dlabels.shape[1] - w, nLabels))

    for i in range(h / 2, rgb.shape[0] - h / 2):
        for j in range(w / 2, rgb.shape[1] - w / 2):
            for l in dlabels[i - h / 2 : i + h / 2, j - w / 2 : j + w / 2].flatten():
                data[i - h / 2, j - w / 2, l] += 1.0

    return data

out = buildDochists(labels2, 64)
#%%
docs = out.reshape((out.shape[0] * out.shape[1], out.shape[2]))
#%%
idf = []
for i in range(64):
    t = 1.0
    for d in docs:
        t += d[i]
    idf.append(t)

idf = len(docs) / numpy.array(idf)

tfidf = []
for d in docs:
    tfidf.append(d * idf)

tfidf = numpy.array(tfidf)
#%%
cimp, _, rank = zip(*sorted(zip(kmeans.cluster_centers_, idf, range(len(idf))), key = lambda x : x[1]))

cimp = numpy.array(cimp)

top10 = set(rank[-48:])
#%%
plt.imshow(kmeans.cluster_centers_.reshape((1, 64, 3)), interpolation = 'NONE')
plt.show()
plt.imshow(cimp.reshape((1, 64, 3)), interpolation = 'NONE')
plt.show()
im3 = numpy.reshape([kmeans.cluster_centers_[i] if i in top10 else [0.0, 0.0, 0.0] for i in kmeans.predict(rgb.reshape((rgb.shape[0] * rgb.shape[1], 3)))], rgb.shape).astype('uint8')
plt.imshow(im3)
plt.gcf().set_size_inches((10, 6))
plt.show()
plt.imshow(numpy.log(idf).reshape((1, idf.shape[0])), cmap = plt.cm.gray, interpolation = 'NONE')
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