#%%
import imageio
import os
import numpy
import time
import json
import skimage.io

os.chdir('/home/bbales2/lineage_process')

import features

vid = imageio.get_reader('saturday_april_23_2.mp4', 'ffmpeg')

frames = []
for im in vid:
    frames.append(im[:360])
    #plt.imshow(frames[0])
    #plt.show()
#%%

try:
    os.mkdir('may_1_annotate')
    os.mkdir('may_1_annotate/ref')
except:
    pass

for i in range(0, len(im), 50):
    skimage.io.imsave('may_1_annotate/ref/frame.{0}.png'.format(i), frames[i])