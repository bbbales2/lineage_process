#%%
import re
import os
import urllib2
import subprocess
import pickle

os.chdir('/home/bbales2/lineage_process/')

f = open('maplist.txt')

lines = []
for line in f.readlines():
    lines.extend(re.sub('</a>', '</a>\n', line).split('\n'))
   
f.close()
#%%

maps = []

for line in lines:
    url = re.search("href=\"(.+)\"", line)
    name = re.search("<a .+>(.+)</a>", line)
    
    if url and name:
        url = url.group(1)
        name = name.group(1)
        
        if re.search("img", url) or re.search("img", name):
            continue
        
        maps.append((url, name, urllib2.urlopen('http://lin1kore.com/www.lineagedb.com/loc/{0}'.format(url)).read()))
        print url
        print '----'
        print name
        print '===='

#%%

f = open('locations.json', 'w')
pickle.dump(maps, f)
f.close()

#%%

location2mop = {}

monsters = {}

for url, name, src in maps:
    location2mop[name] = []
    for murl, mname, level in re.findall("<a href=\"(\.\./monster.+)\">(.+)\(([0-9]+)\)</a>", src):
        location2mop[name].append(mname.strip())
        #monsters[mname.strip()] = { 'url' : murl }
    
#%%
for i, monster in enumerate(monsters):
    monsters[monster]['src'] = urllib2.urlopen('http://lin1kore.com/www.lineagedb.com/loc/{0}'.format(monsters[monster]['url'])).read()
    print i
#%%

os.chdir('/home/bbales2/lineage_process/monsters')    
    
for i, monster in enumerate(monsters):
    monsters[monster]['images'] = []
    for url in re.findall("<img src=\"(\.\./images/monster/.+)\"", monsters[monster]['src']):
        h = subprocess.Popen("wget http://lin1kore.com/www.lineagedb.com/loc/{0}".format(url).split())
        h.communicate()
        
        monsters[monster]['images'].append(url.split('/')[-1])
        
    print i
    
os.chdir('/home/bbales2/lineage_process/')

#%%

f = open('monsters.pkl', 'w')
pickle.dump((location2mop, monsters), f)
f.close()

#%%
<a href="../monster/detail326c.html?id=618&amp;sg=1">Oum Warrior (42)</a>