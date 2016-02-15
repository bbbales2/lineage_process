#%%
import skimage.io
import os
import matplotlib.pyplot as plt
import numpy
import skimage.exposure
import skimage.transform
import skimage.feature

os.chdir('/home/bbales2/lineage_process')

names = sorted(os.listdir('/home/bbales2/lineage_process/screencast_frames'))

im = skimage.io.imread(os.path.join('screencast_frames/', names[47]))

ims = []
ys = []
f = open('labeled.txt')
for line in f:
    line = line.strip()    
    
    if line == '':
        continue

    data = line.split(' ')

    fname = data[2]    
    
    if len(data) == 3:
        cat = 's'
    else:
        cat = data[3]
    
    ys.append(cat)
    
    ims.append(skimage.io.imread(fname))
f.close()

#%%

def get_features(im):
    data = []
    
    for i in range(3):
        data.extend(numpy.histogram(im[:, :, i], range = [0, 255])[0])
    
    return data

Xs = []

for im in ims:
    Xs.append(get_features(im))

import sklearn.linear_model

lr = sklearn.linear_model.LogisticRegression()

sklearn.cross_validation.cross_val_score(lr, Xs, ys)

Xtrain, Xtest, ytrain, ytest = sklearn.cross_validation.train_test_split(Xs, ys, test_size=0.4)
lr.fit(Xtrain, ytrain)

lr.fit(Xtrain, ytrain)

zip(lr.predict(Xtest), ytest)

#%%

import sklearn.cluster

kmeans = sklearn.cluster.KMeans(4)

kmeans.fit(Xs)

#%%

im = skimage.io.imread(os.path.join('screencast_frames/', names[47]))
ax = plt.gca()

w = 40
hists = []

colors = { 's' : 'y',
          'e' : 'w',
          'g' : 'g',
          'k' : 'r' }

plt.imshow(im, interpolation = 'NONE')

for i in range(0, 360, w):
    for j in range(0, im.shape[1], w):
        t = im[i : i + w, j : j + w, :]
                
        cat = kmeans.predict([get_features(t)])[0]
        
        if cat in colors:
            color = colors[cat]
        else:
            color = 'b'
                
        c0 = ax.text(j + w / 2, i + w / 2, cat, horizontalalignment='center', verticalalignment='center')
        c = plt.Rectangle((j, i), w - 1, w - 1, color = color, linewidth=1, fill=False)
        # ax.add_patch(c0)
        ax.add_patch(c)
plt.show()
        
