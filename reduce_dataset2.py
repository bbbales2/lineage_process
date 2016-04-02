#%%
import os
import re
import numpy
import matplotlib
import json
import itertools
import skimage.io

os.chdir('/home/bbales2/lineage_process')

#%%

f = open('images_flat.json')
data = json.load(f)
f.close()

for m in data:
    data[m] = list(itertools.chain(*data[m].values()))

#%%
processed = set()
#%%

for i, m in enumerate(data.keys()):
#m = '2420'
#if True:
    #if len(data[m]) < 100:
    #    continue

    shapes = []
    colors = []
    images = []

    if os.path.exists('all_sprites_160x160/{0}'.format(m)) or m in processed:
        processed.add(m)
        continue

    for f in data[m]:
        try:
            im = skimage.io.imread(f)

            if im.shape[0] > 160 or im.shape[1] > 160:
                continue

            w = im.shape[1]
            h = im.shape[0]

            target = numpy.zeros((160, 160, 4)).astype('uint8')
            target[:, :, 3] = 255
            target[80 - h / 2 : 80 + (h + 1) / 2, 80 - w / 2 : 80 + (w + 1) / 2] = im

            images.append(target)
            colors.append(len(set([tuple(a) for a in target.reshape([target.shape[0] * target.shape[1], 4])])))
            shapes.append(im.shape)

            #skimage.io.imsave('toTrain/{0}.png'.format(len(labels)), target)
            #images.append(target[:, :, :3])
            #files.append('toTrain/{0}.png'.format(len(labels)))
            #labels.append(int(folder.split('/')[1]))
            #folders.append(folder)
        except Exception as e:
            print m, f, e
            pass

    if len(colors) < 50:
        print 'm: ', m
        print 'Insufficient data'
    else:
        print 'm: ', m
        print 'Avg # colors: ', numpy.mean(colors)
        print 'Avg. shape: ', numpy.mean(shapes, axis = 0)
        print '# frames: ', len(images)
        print '# total frames: ', len(data[m])

        images = numpy.array(images)

        images.dump('all_sprites_160x160/{0}'.format(m))

        f = open('all_sprites_160x160_index/{0}.index'.format(m), 'w')
        json.dump( { 'shapes' : shapes, 'colors' : colors }, f )
        f.close()

    processed.add(m)

    print '{0}/{1}'.format(i, len(data))
    print ''
    sys.stdout.flush()
