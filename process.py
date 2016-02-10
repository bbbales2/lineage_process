#%%

import skimage.feature
import matplotlib.pyplot as plt
import skimage.io
import os

os.chdir('/home/bbales2/lineage/')

for i, f in enumerate(sorted(os.listdir('frames_slow'))[100:101]):
    im = skimage.io.imread(os.path.join('frames_slow', f))
    plt.imshow(im)
    plt.show()
    print i

#%%
im = im[:370, :]

hsv = skimage.color.rgb2hsv(im)

imgray = skimage.color.rgb2gray(im)

colors = ['r', 'g', 'b']
plt.imshow(im, interpolation='nearest')
for i in range(3):
    blobs = skimage.feature.blob_dog(hsv[:, :, i], threshold=.1, min_sigma = 10, max_sigma = 30)

    ax = plt.gca()
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=colors[i], linewidth=1, fill=False)
        ax.add_patch(c)

plt.show()