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


def getNewHistogram(histogram,map):
    level = 256
    newHist = np.zeros(level)
    for i in range(level):
        newHist[int(map[i])] += histogram[i]
    return newHist

def getHistogram(im):
    [height,weight] = np.shape(im)
    histogram = np.zeros(256)
    for i in range(height):
        for j in range(weight):
            histogram[int(im[i,j])] += 1
    return histogram / height / weight

def getHistogramMap(frequancyHistogram):
    level = 256
    ac = 0.
    maps = np.zeros(level)
    for i in range(level):
        ac = ac + frequancyHistogram[i]
        maps[i] = np.round((level - 1) * ac)

    return maps

def histogramEqualized(im):
    [height, weight] = np.shape(im)
    histogram = getHistogram(im)
    map = getHistogramMap(histogram)
    newIm = np.zeros([height,weight])
    for i in range(height):
        for j in range(weight):
            newIm[i,j] = map[im[i,j]]

    return newIm

def getMatchingMap(im,targetHist):
    level = 256
    [height, weight] = np.shape(im)
    hist = getHistogram(im)
    map1 = getHistogramMap(hist)
    hist = getNewHistogram(hist,map1)
    map2 = getHistogramMap(targetHist)
    targetHist = getNewHistogram(targetHist,map2)

    map = np.zeros(level)
    sk = 0
    for i in range(level):
        d = np.inf
        sk += hist[i]
        zk = 0
        for j in range(level):
            zk += targetHist[j]
            newd = abs(np.round((level - 1) * (sk)) - np.round((level - 1) * (zk)))
            if  newd <= d:
                d = newd
                map[i] = j
    return map

def histogramMatching(im,targetHist):
    level = 256
    map = getMatchingMap(im,targetHist)
    [height,weight] = np.shape(im)
    for i in range(height):
        for j in range(weight):
            im[i,j] = map[int(im[i,j])]
    return im

def filter2d(im, filter):
    [height,weight] = np.shape(im)
    [h,w] = np.shape(filter)
    newIm = np.zeros([height,weight])
    for i in range(height):
        for j in range(weight):
            sum = 0
            for k in range(h):
                for l in range(w):
                    posi,posj = i + k - h / 2,j + l - w / 2
                    if posi < 0 or posi >= height or\
                        posj < 0 or posj >= weight:
                        continue
                    sum += im[posi,posj] * filter[h - k - 1,w - l - 1]
                newIm[i,j] = sum
    return newIm

def testFilter():
    filename = os.path.join('..', 'hw2_input', '97.png')
    im = imread(filename)
    [height, weight] = np.shape(im)

    plt.subplot(331)
    plt.imshow(im, 'gray')
    plt.title('origin', fontproperties='SimHei')

    filter = np.ones([3,3]) / 9
    newIm = filter2d(im,filter)
    plt.subplot(332)
    plt.imshow(newIm, 'gray')
    plt.title('3*3 mean filter', fontproperties='SimHei')

    filter = np.ones([7,7]) / 49
    newIm = filter2d(im,filter)
    plt.subplot(333)
    plt.imshow(newIm, 'gray')
    plt.title('7*7 mean filter', fontproperties='SimHei')

    filter = np.ones([11,11]) / 121
    newIm = filter2d(im,filter)
    plt.subplot(334)
    plt.imshow(newIm, 'gray')
    plt.title('11*11 mean filter', fontproperties='SimHei')


    laplacian = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    newIm = filter2d(im, laplacian)
    newIm[newIm > 255] = 255
    newIm[newIm < 0] = 0
    plt.subplot(335)
    plt.imshow(newIm, 'gray')
    plt.title('3*3 laplacian filter', fontproperties='SimHei')

    newIm = im + newIm
    newIm[newIm > 255] = 255
    newIm[newIm < 0] = 0
    plt.subplot(336)
    plt.imshow(newIm, 'gray')
    plt.title('shapened image', fontproperties='SimHei')

    plt.show()

def hw2():
    '''
    filename = os.path.join('../', 'hw2_input', '97.png')
    im = imread(filename)
    plt.subplot(231)
    print im,type(im[0,0])
    plt.imshow(im, 'gray')
    plt.title('origin', fontproperties='SimHei')

    plt.subplot(232)
    hist = getHistogram(im)
    plt.bar(np.linspace(0, 256, 256, endpoint=False), \
            hist, alpha=.8, color='r')
    plt.title('origin histogram', fontproperties='SimHei')

    newIm = histogramEqualized(im)
    hist = getHistogram(newIm)
    plt.subplot(233)
    plt.imshow(newIm, 'gray')
    plt.title('equalized image', fontproperties='SimHei')

    plt.subplot(234)
    plt.bar(np.linspace(0, 256, 256, endpoint=False), \
            hist, alpha=.8, color='b')
    plt.title('histogram', fontproperties='SimHei')

    plt.subplot(235)
    newIm = histogramEqualized(newIm)
    newHist = getHistogram(newIm)
    plt.bar(np.linspace(0, 256, 256, endpoint=False), \
            newHist, alpha=.8, color='b')
    plt.title('twice equalization', fontproperties='SimHei')
    print np.max(np.abs(hist - newHist))
    plt.show()
    '''
    testFilter()
if __name__ == '__main__':
    hw2()