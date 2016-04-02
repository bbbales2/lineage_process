#%%

import os
import re
import json
import itertools

os.chdir('/home/bbales2/lineage_process')

folders = [os.path.join('sprite_dump', f) for f in os.listdir('sprite_dump')] + \
            [os.path.join('sprite_dump2', f) for f in os.listdir('sprite_dump2')]

#%%

data = {}

for folder in folders:
    if os.path.isdir(folder):
        data[folder] = { 'animations' : {} }
        count = 0
        for f in os.listdir(folder):
            match = re.match('([A-Za-z0-9]+)-([0-9]+).png', f)

            anim, frame = match.groups()

            anim = anim
            frame = int(frame)

            if anim not in data[folder]['animations']:
                data[folder]['animations'][anim] = []

            data[folder]['animations'][anim].append((frame, f))
            count += 1

        for anim in data[folder]['animations']:
            data[folder]['animations'][anim] = [f for frame, f in sorted(data[folder]['animations'][anim], key = lambda x : x[0])]

        data[folder]['frames'] = count


#%%

for folder in data:
    data[folder] = list(itertools.chain(*map(lambda x : x[1], data[folder]['animations'].items())))

f = open('images2.json', 'w')
json.dump(data, f)
f.close()
#%%

try:
    os.mkdir('all_sprites')
except:
    pass

data4 = {}

for i, folder in enumerate(data):
    base, t = folder.split('/')

    if not os.path.exists('all_sprites/{0}'.format(t)):
        os.mkdir('all_sprites/{0}'.format(t))

    for f in os.listdir(folder):
        try:
            os.symlink('/home/bbales2/lineage_process/{0}/{1}'.format(folder, f), 'all_sprites/{0}/{1}'.format(t, f))

            a = f.split('-')[0]

            if t not in data4:
                data4[t] = {}

            if a not in data4[t]:
                data4[t][a] = []

            data4[t][a].append('all_sprites/{0}/{1}'.format(t, f))
        except Exception as e:
            print e
            print "Error with {0} {1}".format(folder, t)
            pass

    print '{0}/{1}'.format(i, len(data))



#%%

import json

f = open('images_flat.json', 'w')
json.dump(data4, f)
f.close()
#%%

data2 = {}

for folder in data:
    if data[folder]['frames'] > 100:
        data2[folder] = []
        for anim in data[folder]['animations']:
            data2[folder].extend(data[folder]['animations'][anim])

#%%
import skimage.io

data3 = {}

ws = []
hs = []

for folder in data2:
    for frame in data2[folder]:
        im = skimage.io.imread(os.path.join(folder, frame))

        print folder, im.shape

        cvg = float(sum((im[:, :, 3] < 255).flatten())) / (im.shape[0] * im.shape[1])

        #plt.imshow(im[:, :, 3])
        #plt.colorbar()
        #plt.show()

        break

    if cvg > 0.1 and im.shape[0] <= 160 and im.shape[1] <= 160:
        hs.append(im.shape[0])
        ws.append(im.shape[1])

        data3[folder] = data2[folder]

#%%
files = []
folders = []
labels = []
images = []

for folder in data3:
    for frame in numpy.random.choice(data3[folder], 20):
        im = skimage.io.imread(os.path.join(folder, frame))

        if im.shape[0] <= 160 and im.shape[1] <= 160:
            try:
                w = im.shape[1]
                h = im.shape[0]

                target = numpy.zeros((160, 160, 4)).astype('uint8')
                target[:, :, 3] = 255
                target[80 - h / 2 : 80 + (h + 1) / 2, 80 - w / 2 : 80 + (w + 1) / 2] = im

                if skimage.color.rgb2gray(target).flatten().mean() < 0.01:
                    continue
            except:
                pass

            #skimage.io.imsave('toTrain/{0}.png'.format(len(labels)), target)
            images.append(target[:, :, :3])
            files.append('toTrain/{0}.png'.format(len(labels)))
            labels.append(int(folder.split('/')[1]))
            folders.append(folder)
    print folder

#%%
images = numpy.array(images)

f = open('totrain.json', 'w')
json.dump((files, folders, labels), f)
f.close()

images.dump('toTrain.dat')

#%%
import skimage.color
#%%
a = skimage.color.rgb2gray(skimage.io.imread('toTrain/5031.png'))
#%%

import json

f = open('images2.json', 'w')
json.dump(data3, f)
f.close()