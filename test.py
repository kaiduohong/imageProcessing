#from skimage import data
import skimage as skm
import numpy as np
import pylab as pl
from scipy import misc
from matplotlib import pyplot as plt

cat = misc.face()
cat = np.zeros(1000000).reshape(100,100) + 1
reddish = cat[:, :,0] > 160
cat[reddish] = [test.py:150, 255, 0]
print cat
print np.shape(cat)
print cat[0,0,2]

plt.imshow(cat)
plt.show()
pl.imshow(cat)