#%%

import skimage.io
import os
import matplotlib.pyplot as plt
import numpy
import mahotas.features.surf
import skimage.exposure
import skimage.transform
import skimage.feature
import scipy
import time

import sklearn.cluster


colors = ['b',
          'g',
          'r',
          'c',
          'm',
          'y',
          'k',
          'w']
os.chdir('/home/bbales2/lineage_process')

names = sorted(os.listdir('/home/bbales2/lineage_process/screencast_frames'))

im = skimage.io.imread(os.path.join('screencast_frames/', names[33]))

#%%

import rfs_filters

reload(rfs_filters)

filters = rfs_filters.make(25, [1])

#%%
print time.time()

im = skimage.io.imread(os.path.join('screencast_frames/', names[33]))
#im = skimage.transform.rescale(im, 0.5)[:180, :]

plt.imshow(im)
plt.show()
filtered = numpy.zeros((im.shape[0], im.shape[1], filters.shape[2]))
for i in range(filters.shape[2]):
    filtered[:, :, i] = scipy.signal.fftconvolve(im[:, :, 0], filters[:, :, i], mode = 'same')

#filtered2 = numpy.zeros((im.shape[0], im.shape[1], 8))
#for i in range(0, 6):
#    filtered2[:, :, i] = numpy.max(filtered[:, :, i * 6 : (i + 1) * 6], axis = 2)

filtered = filtered.reshape(im.shape[0] * im.shape[1], filters.shape[2])

print time.time()
#%%
import sklearn.cluster

cl = sklearn.cluster.KMeans(10)

pred = cl.fit_predict(filtered2)

pred = pred.reshape(im.shape[0:2])

plt.imshow(pred)
plt.show()
#%%
tmp = time.time()
output = skimage.feature.local_binary_pattern(im[:, :, 0], 8, 11)
print time.time() - tmp

plt.imshow(output, cmap = plt.cm.gray)
#%%
for fname in names[0:40]:
    im = skimage.io.imread(os.path.join('screencast_frames/', fname))

    descriptors, vis = skimage.feature.daisy(im[:370, :, 0], visualize = True)
    plt.imshow(im)
    plt.show()

    plt.imshow(vis)
    plt.show()
    1/0
#%%
for fname in names[0:40]:
    im = skimage.io.imread(os.path.join('screencast_frames/', fname))
    im = skimage.transform.rescale(im, 0.5)[25:180, :]
    image = numpy.array(im[:, :, 0])
    print time.time()
    fd, hog_image = skimage.feature.hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    print time.time()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)

    #ax1.axis('off')
    #ax1.imshow(image, cmap=plt.cm.gray)
    #ax1.set_title('Input image')
    #ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    #plt.imshow(im)

    hir = skimage.exposure.rescale_intensity(hog_image, in_range = (0, 100.0))
    ax1.imshow(im)
    ax2.imshow(hir)
    #ax.imshow(hog, interpolation = 'NONE', clim = [0.01, 1.0])
    plt.show()
#%%

for fname in names[0:40]:
    im = skimage.io.imread(os.path.join('screencast_frames/', fname))
    im = skimage.transform.rescale(im, 0.5)

    cim = im[:170, :, :]
    ctotal = numpy.sum([skimage.feature.canny(cim[:, :, i], 3.0) for i in range(3)], axis = 0).astype('float')

    png3 = -ctotal#skimage.color.rgb2gray(1 - output)

    #png3 = skimage.transform.rescale(png3, 0.5)[:170, :]
    print time.time()
    blobs_dog = skimage.feature.blob_dog(1 - png3, min_sigma = 5, max_sigma = 15, threshold = .05, overlap = 1.0)
    blobs_dog[:, 2] = blobs_dog[:, 2] * numpy.sqrt(2)
    print time.time()

    plt.imshow(im, interpolation='NONE')
    blobs = blobs_dog
    ax = plt.gca()
    for point in blobs:
        y = point[0]
        x = point[1]
        r = point[2]
        c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
        ax.add_patch(c)
    plt.show()
#%%
descs, descs_image = skimage.feature.daisy(im[:, :, 0], visualize = True)
#%%
censure = skimage.feature.CENSURE(1, 15)
#%%
points = censure.detect(im[:, :, 0])
#%%
kmeans = sklearn.cluster.KMeans(4)

#%%
kmeans.fit(descriptors)
#%%
orb = skimage.feature.ORB(n_keypoints = 500, n_scales = 8, downscale = 1.2)

descriptors = []

for fname in names[0:40]:
    im = skimage.io.imread(os.path.join('screencast_frames/', fname))
    im3 = skimage.transform.rescale(im[:370, :], 0.25)
    im2 = skimage.util.pad(im3[:, :], [(20, 20), (20, 20), (0, 0)], 'reflect')
    im2 = skimage.color.rgb2hsv(im2)
    #plt.imshow(im2)
    #plt.show()

    print time.time()
    orb.detect_and_extract(im2[:, :, 2])
    print time.time()

    descriptors.extend(orb.descriptors.astype('float'))

    plt.imshow(im, interpolation='NONE')
    ax = plt.gca()
    for (y, x), desc in zip(orb.keypoints, orb.descriptors):
        color = colors[kmeans.predict([desc.astype('float')])[0]]
        c = plt.Circle((4 * (x - 20), 4 * (y - 20)), 5, color = color, linewidth=1, fill=False)
        ax.add_patch(c)
    plt.show()

#%%
#%%
corners = skimage.filters.sobel(im[:370, :, 0])

plt.imshow(corners, cmap = plt.cm.gray)
#%%

corners = skimage.feature.corner_fast(im[:370, :, 0], n = 5, threshold= 0.15)

plt.imshow(corners, cmap = plt.cm.gray)
#%%
print time.time()
w = 40
hists = []
for oi in [0]:
    for oj in [0]:
        for i in range(0, 360, w):
            for j in range(0, im.shape[1], w):
                plt.imshow(im, interpolation = 'NONE')
                c = plt.Rectangle((j, i), w - 1, w - 1, color=random.choice(['r', 'g', 'b']), linewidth=1, fill=False)
                ax.add_patch(c)
                plt.show()

                a = raw_input("Classify: ")

                skimage.io.imsave(os.path.join('training', 'im_{2:0>3}_{0:0>3}_{3:0>3}_{1:0>3}.png'.format(oi, oj, i, j)), im[i + oi : i + w + oi, j + oj : j + w + oj, :])
        #hists.append(skimage.exposure.histogram(im[i : i + w, j : j + w, 0].astype('float'), 32))
print time.time()

#%%
#plt.imshow(ctotal, interpolation='NONE', cmap = plt.cm.gray)
plt.imshow(im, interpolation = 'NONE')
blobs = blobs_dog
ax = plt.gca()
for i in range(0, 360, w):
    for j in range(0, im.shape[1], w):
        c = plt.Rectangle((j, i), w - 1, w - 1, color=random.choice(['r', 'g', 'b']), linewidth=1, fill=False)
        ax.add_patch(c)
plt.show()