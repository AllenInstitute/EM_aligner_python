import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#width and height of final image
w = 3840
h = 3840
n=16
overlap = w/n

full = np.zeros((h,w)).astype('uint8')

#make some lines on top and left edge
lw = 4
nlines = 8
space = (overlap-nlines*lw)/nlines-1
for i in np.arange(nlines+1):
    edge = (i+1)*space
    full[:,edge:edge+lw] = 255
for i in np.arange(1,n):
    full[:,i*overlap:(i+1)*overlap] = full[:,0:overlap]

#reflect top edge to left side
full = full + np.rot90(np.fliplr(full),1)

#make mark to break the symmetries
full[1100:1400,1100:1200] = full[1100:1400,1100:1200]+255
full[1100:1200,1100:1400] = full[1100:1200,1100:1400]+255
#mark the center
csize = 500
full[(w/2-csize):(w/2+csize),(w/2-csize):(w/2+csize)] = 255

p = Image.fromarray(full)
p.save('test_pattern.tif')

plt.figure(1)
plt.clf()
plt.subplot(1,2,1)
plt.imshow(full,cmap='gray_r',interpolation='nearest')
plt.subplot(1,2,2)
plt.imshow(full,cmap='gray_r',interpolation='nearest',extent=[0,w,h,0])
plt.imshow(full,cmap='gray_r',interpolation='nearest',extent=[w-overlap,2*w-overlap,h,0])
plt.imshow(full,cmap='gray_r',interpolation='nearest',extent=[0,w,2*h-overlap,h-overlap])
plt.imshow(full,cmap='gray_r',interpolation='nearest',extent=[w-overlap,2*w-overlap,2*h-overlap,h-overlap])

plt.gca().set_xlim(0,2*w)
plt.gca().set_ylim(2*h,0)

