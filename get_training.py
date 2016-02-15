import skimage.io
import os
import matplotlib.pyplot as plt
import numpy
import skimage.exposure
import skimage.transform
import skimage.feature
import json

os.chdir('/home/bbales2/lineage_process')

names = sorted(os.listdir('/home/bbales2/lineage_process/screencast_frames'))

im = skimage.io.imread(os.path.join('screencast_frames/', names[47]))

w = 40
hists = []

data = []

for i in range(0, 360, w):
    for j in range(0, im.shape[1], w):
        plt.imshow(im, interpolation = 'NONE')
        c = plt.Rectangle((j, i), w - 1, w - 1, color = 'r', linewidth=1, fill=False)
        ax = plt.gca()
        ax.add_patch(c)
        plt.show()
                
        a = raw_input("Classify: ")                
            
        if a == 'stop':
            break;
        print a                

        fname = os.path.join('training', 'im_{0:0>3}_{1:0>3}.png'.format(i, j))

        data.append((i, j, fname, a))

        skimage.io.imsave(fname, im[i : i + w, j : j + w, :])

    if a == 'stop':
        break
    
f = open('labeled.txt', 'w')
for i, j, fname, a in data:
    f.write("{0} {1} {2} {3}\n".format(i, j, fname, a))
f.close()
