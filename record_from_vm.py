#%%
import virtualbox
import tempfile
import skimage.io
import matplotlib.pyplot as plt
import time
import array
import sys
import numpy
#%%

vbox = virtualbox.VirtualBox()

session = virtualbox.Session()

vm = vbox.find_machine('Windows')

session = vm.create_session()

cId = None

#%%
def get_screenshot(png, h, w):
    a = array.array('B')
    a.fromstring(png)
    data = numpy.array(list(a)).reshape(w, h, 4, order = 'C')
    data = data[:, :, :3].astype('uint8')

    return data

h, w, _, _, _ = session.console.display.get_screen_resolution(0)

#tmp = time.time()
#data = get_screenshot(h, w)
#print time.time() - tmp

#plt.imshow(data)
#plt.show()
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
import pickle

f = open('recorded.pkl', 'w')
screens = numpy.array([get_screenshot(screen, h, w) for screen in screens])

numpy.save('recorded.numpy', screens)

plt.imshow(screens[0])
plt.show()

pickle.dump(presses, f)
f.close()
#exit(1)
#%%
#for (j, i), screen in zip(presses, screens):
#    plt.imshow()
#    circle = plt.Circle((j, i), 10, color = 'r', fill = False)

#    plt.gca().add_artist(circle)
#    plt.show()
#%%

