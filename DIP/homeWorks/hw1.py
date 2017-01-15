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
    plt.imshow(im, mpl.cm.gray_r)
    plt.title('origin', fontproperties='SimHei')

    levels = [128, 32, 8, 4, 2]
    for (i,level) in enumerate(levels):
        newIm = quantize(im, level)
        plt.subplot(232+i)
        plt.imshow(newIm, mpl.cm.gray_r)
        plt.title('level='+str(level), fontproperties='SimHei')

    plt.show()

    plt.subplot(231)
    plt.imshow(im, mpl.cm.gray_r)
    plt.title('origin', fontproperties='SimHei')

    plt.subplot(232)
    hist = getHistogram(im)
    plt.bar(np.linspace(0, 256, 256, endpoint=False), \
            hist, alpha=.8, color='r')

    [hist, newIm] = histogramEqualized(im)
    plt.subplot(233)
    plt.imshow(newIm, mpl.cm.gray_r)
    plt.title('origin', fontproperties='SimHei')


    plt.subplot(234)
    plt.bar(np.linspace(0, 256, 256, endpoint=False), \
            hist, alpha=.8, color='g')


    plt.subplot(235)
    [newHist, newIm] = histogramEqualized(newIm)
    plt.bar(np.linspace(0, 256, 256, endpoint=False), \
            newHist, alpha=.8, color='g')
    plt.show()


if __name__ == '__main__':
    #hw1()