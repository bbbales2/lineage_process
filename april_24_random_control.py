#%%

#import imageio
import os
import numpy
import time
import json
import bisect
import skimage.filters
import skimage.color
import virtualbox
import array
import skimage.transform

os.chdir('/home/bbales2/lineage_process')

import features

#%%

vbox = virtualbox.VirtualBox()

session = virtualbox.Session()

vm = vbox.find_machine('Windows')

session = vm.create_session()

cId = None

#%%

def get_screenshot(h, w):
    png = session.console.display.take_screen_shot_to_array(0, h, w)
    data = numpy.array(bytearray(png)).reshape(w, h, 4, order = 'C')
    data = data[:, :, :3].astype('uint8')

    return data

h, w, _, _, _ = session.console.display.get_screen_resolution(0)
#%%
plt.imshow(get_screenshot(640, 480))
#%%
dt = 0.2

im = None
dx = None
dx2 = None

Xs = None
#plt.imshow(inf)

lastClick = 0.0

while 1:
    start = time.time()
    im2 = get_screenshot(640, 480).astype('double')[:360, :640] / 255.0

    if im != None:
        dx2 = im - im2
        
    if dx2 != None:
        dx2 = numpy.linalg.norm(features.rgbhist(dx2, 16), axis = 2)
        
        if Xs == None:
            Xs, Ys = numpy.meshgrid(numpy.arange(dx2.shape[1]), numpy.arange(dx2.shape[0]))
            Rs = (Xs - dx2.shape[1] / 2)**2 / 2.0 + (Ys - dx2.shape[0] / 2)**2
            Rs2 = (Xs - dx2.shape[1] / 2)**2 / 1.5 + (Ys - dx2.shape[0] / 2)**2 / 2.0
            inf = numpy.exp(-Rs / 100.0) - numpy.exp(-Rs2 / 5.0)
        
        if dx == None:
            dx = dx2
        else:
            dx = 0.5 * dx + 0.5 * dx2

    if dx != None:
        probs = (inf * dx).flatten()
    
        if False:#numpy.random.rand() < 0.1:
            csum = numpy.cumsum(probs)
    
            choice = numpy.random.rand() * csum[-1]
    
            idx2 = numpy.unravel_index(bisect.bisect_left(csum, choice), dx.shape)
        else:
            idx2 = numpy.unravel_index(numpy.argmax(probs), dx.shape)
    
        idx = numpy.array(idx2)
        
        #print idx
        
        
        if time.time() - lastClick > 5.0:
            x = idx[1] * im2.shape[1] / dx2.shape[1] + im2.shape[1] / dx2.shape[1] / 2
            y = idx[0] * im2.shape[0] / dx2.shape[0] + im2.shape[0] / dx2.shape[0] / 2

            #x = 320
            #y = 220            
            
            session.console.mouse.put_mouse_event_absolute(x, y, 0, 0, 0)
            time.sleep(0.001)
            session.console.mouse.put_mouse_event_absolute(x, y, 0, 0, 1)
            #time.sleep(0.000)
            #session.console.mouse.put_mouse_event_absolute(x, y, 0, 0, 0)
            for x2, y2 in zip(numpy.linspace(x, 320.0, 10), numpy.linspace(y, 440, 10)):
                session.console.mouse.put_mouse_event_absolute(int(x2), int(y2), 0, 0, 1)
                time.sleep(0.005)
            print numpy.mean(dx)
            print x, y
                
            session.console.mouse.put_mouse_event_absolute(320, 440, 0, 0, 0)


            plt.imshow(im2)
            plt.imshow(inf * dx, interpolation = 'NONE', alpha = 0.5, extent = (0, im2.shape[1], im2.shape[0], 0))
            plt.show()
            #session.console.mouse.put_mouse_event_absolute(320, 600, 0, 0, 1)
            #time.sleep(0.005)
            
            lastClick = time.time()
    
    im = im2
    sleep = max(0.0, 0.2 - (time.time() - start))
    print "Sleeping (ms) ", sleep * 1000
    time.sleep(sleep)
    #plt.imshow(im)
    #%%
tmp = time.time()
t = numpy.array(bytearray(png)).reshape(w, h, 4)
print time.time() - tmp
#%%
print [26, 38]
print im2.shape[1] / dx.shape[1]

#0.00315801744418
#308 212

#%%
lastButton = 0
presses = []
screens = []

def test(a):
    global lastButton
    global presses

    try:
        newButton = int(a.buttons)
    except:
        newButton = 0

    if lastButton == 0 and newButton == 1:
        presses.append((a.x, a.y))
        screens.append(session.console.display.take_screen_shot_to_array(0, h, w))

    lastButton = newButton

mouse_events = session.console.mouse.event_source

if cId:
    virtualbox.events.unregister_callback(cId)

cId = virtualbox.events.register_callback(test, mouse_events, virtualbox.library.VBoxEventType.on_guest_mouse)

try:
    while 1:
        time.sleep(0.1)
        sys.stdout.flush()
except:
    pass
#%%

    dx2.append(numpy.linalg.norm(features.rgbhist(dx, 16), axis = 2))

#%%
im = dx2[0]
ims = []
#%%
Xs, Ys = numpy.meshgrid(numpy.arange(im.shape[1]), numpy.arange(im.shape[0]))
Rs = (Xs - 18.5)**2 / 2.0 + (Ys - 11.5)**2
Rs2 = (Xs - 18.5)**2 / 1.5 + (Ys - 11.5)**2 / 2.0
inf = numpy.exp(-Rs / 100.0) - numpy.exp(-Rs2 / 10.0)
#plt.imshow(inf)

#idx = numpy.array([11.5, 18.5])

for image, im2 in zip(images2, dx2):
    im = im * 0.75 + im2 * 0.25
    #m = numpy.linalg.norm(im, axis = 2)
    im2 = numpy.array(im)
    im2[10:15, 19:20] = 0.0
    probs = (inf * im2).flatten()

    if numpy.random.rand() < 0.1:
        csum = numpy.cumsum(probs)

        choice = numpy.random.rand() * csum[-1]

        idx2 = numpy.unravel_index(bisect.bisect_left(csum, choice), im2.shape)
    else:
        idx2 = numpy.unravel_index(numpy.argmax(probs), im2.shape)

    idx = numpy.array(idx2)#0.75 * idx + 0.25 *

    im2[int(idx[0]), int(idx[1])] = 100.0

    plt.imshow(image)
    plt.imshow(inf * im2, interpolation = 'NONE', alpha = 0.5, extent = (0, image.shape[1], image.shape[0], 0))
    plt.show()
    print 'hi'
    #ims.append(im)
#%%