#-*-coding:utf8-*-
import numpy as np
from scipy.misc import imread, imsave
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import skimage
from skimage import io
import sys

def scale(im,h,w):
    [height,weight] = np.shape(im)
    newIm = np.zeros([h,w],dtype=int)
    downH = downW = True
    if h > height:
        downH = False
    if w > weight:
        downW = False
    for i in range(h):
        for j in range(w):
            [u,posi] = np.modf(1. * i * height / h)
            [v,posj] = np.modf(1. * j * weight / w)
            if downH: u = 0
            if downW: v = 0
            posi = int(posi)
            posj = int(posj)
            a = b = c = d = im[posi, posj]
            if posj + 1 < weight:
                b = im[posi, posj + 1]
            if posi + 1 < height:
                c = im[posi + 1, posj]
            if posi + 1 < height and posj + 1 < weight:
                d = im[posi + 1, posj + 1]
            newIm[i, j] = int(np.round((1 - u) * (1 - v) * a +\
                                   (1 - u) * v * b + u * (1 - v) * c +\
                                   u * v * d))
    return newIm

def quantize(im,level):
    olevel = 256.
    im = im.copy()
    [height,weight] = np.shape(im)
    for i in range(height):
        for j in range(weight):
            im[i,j] = np.uint8(np.round(((olevel - 1) / (level - 1)) * \
                               np.floor(im[i,j] * level / olevel) ))
    return im

def downSampling(im,h,w):
	[height,weight] = np.shape(im)
	newIm = np.zeros([h,w])
	for i in range(h):
		for j in range(w):
 			posi = int(np.floor(i * height / h))
			posj = int(np.floor(j * weight / w))
			newIm[i,j] = im[posi,posj]
	return newIm

def hw1():
    filename = os.path.join('../', 'hw1_input', '97.png')
    im = imread(filename)

    [height,weight] = np.shape(im)

    plt.subplot(331)
    plt.imshow(im, mpl.cm.gray_r)
    plt.title(str(weight)+'*'+str(height),fontproperties='SimHei')

    newIm = scale(im, 128, 192)
    plt.subplot(332)
    plt.imshow(newIm, mpl.cm.gray_r)
    plt.title(u'192×128', fontproperties='SimHei')

    newIm = scale(im, 64, 96)
    plt.subplot(333)
    plt.imshow(newIm, mpl.cm.gray_r)
    plt.title(u'96×64', fontproperties='SimHei')

    newIm = scale(im, 32, 48)
    plt.subplot(334)
    plt.imshow(newIm, mpl.cm.gray_r)
    plt.title(u'48×32', fontproperties='SimHei')


    newIm = scale(im, 16, 24)
    plt.subplot(335)
    plt.imshow(newIm, mpl.cm.gray_r)
    plt.title(u'24×16', fontproperties='SimHei')

    newIm = scale(im, 8, 12)
    plt.subplot(336)
    plt.imshow(newIm, mpl.cm.gray_r)
    plt.title(u'12×8', fontproperties='SimHei')

    newIm = scale(im, 200, 300)
    plt.subplot(337)
    plt.imshow(newIm, mpl.cm.gray_r)
    plt.title(u'300×200', fontproperties='SimHei')

    newIm = scale(im, 300, 450)
    plt.subplot(338)
    plt.imshow(newIm, mpl.cm.gray_r)
    plt.title(u'450×300', fontproperties='SimHei')

    newIm = scale(im, 200, 500)
    plt.subplot(339)
    plt.imshow(newIm, mpl.cm.gray_r)
    plt.title(u'500×200', fontproperties='SimHei')

    plt.show()

    plt.subplot(231)
    print type(im[0,0])
    plt.imshow(im, 'gray')
    plt.title('origin', fontproperties='SimHei')

    levels = [128, 32, 8, 4, 2]

    for (i,level) in enumerate(levels):
        newIm = quantize(im, level)
        plt.subplot(232+i)
        plt.imshow(newIm, 'gray')
        plt.title('level='+str(level), fontproperties='SimHei')

    plt.show()



if __name__ == '__main__':
    hw1()