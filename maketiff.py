import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#width and height of final image
w = 3840
h = 3840

#create 1/4 sub-image for preserving x/y reflection symmetry in final image
w4 = w/2
h4 = h/2
pattern = np.zeros((h4,w4)).astype('uint8')

#make some horizontal lines
nlines = 10
wsmall = 4
wstep = 2
centerstart = 10 #pixels from edge for first line
for i in np.arange(nlines):
    wid = wsmall+i*wstep
    c = centerstart+i*h4/(nlines)
    pattern[(c-wid/2):(c+wid/2),:]=255

#reflect around 45deg to make x/y symmetry
pattern = pattern + np.rot90(pattern,3)

full = np.zeros((h,w)).astype('uint8')
#make mark to break the symmetries
full[1100:1920,1100:1920] = 255
full[1200:1920,1200:1920] = 0

order = [1,0,2,3]
for i in np.arange(2):
    for j in np.arange(2):
        full[(i*h4):((i+1)*h4),(j*w4):((j+1)*w4)] = full[(i*h4):((i+1)*h4),(j*w4):((j+1)*w4)]+np.rot90(pattern,order[i*2+j])


p = Image.fromarray(full)
p.save('test_pattern.tif')

plt.figure(1)
plt.clf()
plt.subplot(1,2,1)
plt.imshow(pattern,cmap='gray_r',interpolation='nearest')
plt.subplot(1,2,2)
plt.imshow(full,cmap='gray_r',interpolation='nearest')
