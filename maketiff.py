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
    full[0:2*overlap,edge:edge+lw] = 255
    full[edge:edge+lw,0:2*overlap] = 255
for i in np.arange(1,n/2):
    full[0:2*overlap,i*2*overlap:(i+1)*2*overlap] = full[0:2*overlap,0:2*overlap]

#reflect top edge to left side
full = full + np.rot90(np.fliplr(full),1)

#make the bottom overlap pixels exactly match the top overlap pixels
full[(h-overlap):h,:] = full[0:overlap,:]
#make the right overlap pixels exactly match the left overlap pixels
full[:,(w-overlap):w] = full[:,0:overlap]

#make mark to break the symmetries
full[1100:1920,1100:1920] = 255
full[1200:1920,1200:1920] = 0

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

