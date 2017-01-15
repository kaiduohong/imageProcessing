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
from hw2 import filter2d
from hw2 import histogramEqualized
from hw2 import getHistogram
from hw2 import histogramMatching

#以0到255灰度级输入,输出也是0到255
def rgb2hsi(im):
    [height,weight,dim] =np.shape(im)
    hue = np.zeros([height,weight])
    saturation = np.zeros([height,weight])
    idensity = np.zeros([height,weight])
    im = im.astype(np.float) / 255.

    for i in range(height):
        for j in range(weight):
            [r,g,b] = im[i,j,0:3]
            theta = np.arccos(0.5*((r-g)+(r-b))/\
                              (np.sqrt((r-g)*(r-g) + \
                                (r-b)*(g-b)) + sys.float_info.epsilon))/\
                                (2 * np.pi)
            if b <= g:
                hue[i,j] = theta
            else:
                hue[i,j] = 1 - theta
            saturation[i,j] = 1 - 3 * min([r,g,b]) / (r+g+b+sys.float_info.epsilon)
            idensity[i,j] = (r + g + b) / 3.0

    hsiIm = np.zeros([height,weight,dim])
    hsiIm[:,:,0],hsiIm[:,:,1],hsiIm[:,:,2] =  \
        hue,saturation,idensity

    return (hsiIm * 255.).astype(np.uint8)

#输入是0到255灰度级的
def hsi2rgb(hue,saturation,idensity):
    [height,weight]  = np.shape(hue)
    im = np.zeros([height, weight,3])
    for i in range(height):
        for j in range(weight):
            [h,s,ind] = hue[i,j],saturation[i,j],idensity[i,j]
            h = float(h) / 255.
            s = float(s) / 255.
            ind = float(ind) / 255.
            h *= 2 * np.pi

            if h < 2. / 3 * np.pi:
                b = ind * (1 - s)
                r = ind * (1 + s * np.cos(h) / np.cos(np.pi/3 - h))
                g = 3 * ind - r - b
            elif h < 4. / 3 * np.pi:
                h -= 2 * np.pi / 3
                r = ind * (1 - s)
                g = ind * (1 + s * np.cos(h) / np.cos(np.pi/3 - h))
                b = 3 * ind - r - g
            else:
                h -= 4 * np.pi / 3
                g = ind * (1 - s)
                b = ind * (1 + s * np.cos(h) / np.cos(np.pi / 3 - h))
                r = 3 * ind - b - g
                print h,s,ind,r,g,b
            im[i,j,0],im[i,j,1],im[i,j,2] = r*255,g*255,b*255
    return im.astype(np.uint8)

def testC():
    matrix = np.ones([50,50])
    cyan = np.zeros([50,50,3])
    green = np.zeros([50,50,3])
    magenta = np.zeros([50,50,3])
    yellow = np.zeros([50,50,3])

    yellow[:,:,0] =  yellow[:,:,1] = matrix
    magenta[:,:,0] = magenta[:,:,2] = matrix
    green[:,:,1] = matrix
    cyan[:,:,1] = cyan[:,:,2] = matrix

    im = np.zeros([100,100,3])
    im[:50,:50] = yellow
    im[:50,50:] = magenta
    im[50:,:50] = cyan
    im[50:,50:] = green
    oim = im
    [h,s,i] = rgb2hsi(im)

    plt.subplot(331)
    io.imshow(h)
    plt.subplot(332)
    io.imshow(s)

    #plt.subplot(222)
    #io.imshow(i)
    plt.subplot(333)
    io.imshow(i)

    #plt.subplot(224)
    #io.imshow(h)
    im = hsi2rgb(h,s,i)
    plt.subplot(334)
    io.imshow(im)

    plt.subplot(335)
    io.imshow(oim)

    meanFilter = np.ones([16, 16]) / (16. * 16)

    h = filter2d(h,meanFilter)

    im = hsi2rgb(h,s,i)
    plt.subplot(336)
    io.imshow(im)

    plt.show()



def generateGaussNoise(m,n,mean,sigma):
    np.random.normal()
    return np.random.normal(mean,sigma,size=[m,n])
def addGaussNoise(im,mean=0,sigma=1):
    [height,weight] = np.shape(im)
    noise = generateGaussNoise(height,weight,mean,sigma)
    im = im + noise
    im[im > 255] = 255.
    im[im < 0] = 0.
    return im
def addSaltPepperNoise(im,p,q):
    [m,n] = np.shape(im)
    im = im.copy()
    for i in range(m):
        for j in range(n):
            if np.random.uniform() <= p:
                    im[i,j] = 255
            elif np.random.uniform() <= q:
                    im[i,j] = 0
    return im

def arithmeticMeanFilter(im,m,n):
    filter = np.ones([m,n]) / (m*n)
    return filter2d(im,filter)
def geometricMeanFilter(im,h,w):
    [height,weight] = np.shape(im)
    newIm = np.zeros([height,weight])
    for i in range(height):
        for j in range(weight):
            sum = 1
            count = 0
            for k in range(h):
                for l in range(w):
                    posi, posj = int(i + k - h / 2), int(j + l - w / 2)

                    if posi < 0 or posi >= height or \
                                    posj < 0 or posj >= weight:
                        continue
                    count += 1
                    sum *= im[posi, posj]
            newIm[i, j] = np.power(sum,1. / count)
    return newIm

def medianFiltering(im,m,n):
    [height,weight] = np.shape(im)
    newIm = np.zeros([height, weight])
    for i in range(height):
        for j in range(weight):
            arr = []
            for k in range(m):
                for l in range(n):
                    posi, posj = i + k - m / 2, j + l - n / 2
                    if posi < 0 or posi >= height or \
                                    posj < 0 or posj >= weight:
                        continue
                    arr.append(im[posi, posj])
            newIm[i, j] = np.median(arr)
    return newIm

def minFilter(im,m,n):
    [height, weight] = np.shape(im)
    newIm = np.zeros([height, weight])
    for i in range(height):
        for j in range(weight):
            arr = []
            for k in range(m):
                for l in range(n):
                    posi, posj = i + k - m / 2, j + l - n / 2
                    if posi < 0 or posi >= height or \
                                    posj < 0 or posj >= weight:
                        continue
                    arr.append(im[posi, posj])
            newIm[i, j] = np.min(arr)
    return newIm


def harmonicMeanFilter(im,n,m):
    [height,weight] = np.shape(im)
    newIm = np.zeros([height,weight])
    for i in range(height):
        for j in range(weight):
            count = 0
            sum = 0
            for k in range(n):
                for l in range(m):
                    posi, posj = i + k - n / 2, j + l - m / 2
                    if posi < 0 or posi >= height or \
                                    posj < 0 or posj >= weight:
                        continue
                    count += 1
                    #防止除零
                    sum += 1. / (im[posi,posj]+sys.float_info.epsilon)
            newIm[i,j] = count / sum
    return newIm

def contraharmonicMeanFilter(im,n,m,Q):
    [height, weight] = np.shape(im)
    newIm = np.zeros([height, weight])
    for i in range(height):
        for j in range(weight):
            sum = 0
            sum2 = 0
            for k in range(n):
                for l in range(m):
                    posi, posj = i + k - n / 2, j + l - m / 2
                    if posi < 0 or posi >= height or \
                                    posj < 0 or posj >= weight:
                        continue
                    # 防止除零
                    sum += np.power((im[posi,posj]+sys.float_info.epsilon),Q+1)
                    sum2 += np.power((im[posi,posj]+sys.float_info.epsilon),Q)
            newIm[i, j] = sum / (sum2 + sys.float_info.epsilon)
    return newIm

def task1():
    filename = os.path.join('..', 'hw4_input', 'task_1.png')
    im = imread(filename) / 255.
    tbtMeanFilter = np.ones([3, 3]) / 9.
    nbnMeanFilter = np.ones([9, 9]) / 81.
    im1 = filter2d(im, tbtMeanFilter)
    im2 = filter2d(im, nbnMeanFilter)
    im3 = harmonicMeanFilter(im, 3, 3)
    im4 = harmonicMeanFilter(im, 9, 9)
    im5 = contraharmonicMeanFilter(im, 3, 3, 1.5)
    im6 = contraharmonicMeanFilter(im, 9, 9, 1.5)

    plt.subplot(331); plt.imshow(im , cmap='gray')
    plt.title('origin', fontproperties='SimHei')
    plt.subplot(332); plt.imshow(im1, cmap='gray')
    plt.title('3 * 3 mean filter', fontproperties='SimHei')
    plt.subplot(333); plt.imshow(im2, cmap='gray')
    plt.title('9 * 9 mean filter', fontproperties='SimHei')
    plt.subplot(334); plt.imshow(im3, cmap='gray')
    plt.title('3 * 3 harmonic mean filter', fontproperties='SimHei')
    plt.subplot(335); plt.imshow(im4, cmap='gray')
    plt.title('9 * 9 harmonic mean filter', fontproperties='SimHei')
    plt.subplot(336); plt.imshow(im5, cmap='gray')
    plt.title('3 * 3 contraharmonic mean filter', fontproperties='SimHei')
    plt.subplot(337); plt.imshow(im6, cmap='gray')
    plt.title('9 * 9 contraharmonic mean filter', fontproperties='SimHei')

    plt.show()

def task2():
    filename = os.path.join('..', 'hw4_input', 'task_2.png')
    im = imread(filename)
    im = im[:,:,0]
    plt.subplot(331); plt.imshow(im, cmap='gray')
    plt.title('origin', fontproperties='SimHei')

    im1 = addGaussNoise(im,0,40)
    im2 = addSaltPepperNoise(im,0.2,0.2)
    im3 = addSaltPepperNoise(im,0.2,0)
    im4 = arithmeticMeanFilter(im1,3,3)
    im5 = geometricMeanFilter(im1,3,3)
    im6 = medianFiltering(im1,3,3)


    plt.subplot(332); plt.imshow(im1, cmap='gray')
    plt.title('image with gaussian noise', fontproperties='SimHei')
    plt.subplot(333); plt.imshow(im2, cmap='gray')
    plt.title('image with salt and pepper noise', fontproperties='SimHei')
    plt.subplot(334); plt.imshow(im3, cmap='gray')
    plt.title('image with salt noise', fontproperties='SimHei')

    plt.subplot(335); plt.imshow(im4, cmap='gray')
    plt.title('arithmeticMeanFilter denoised', fontproperties='SimHei')
    plt.subplot(336); plt.imshow(im5, cmap='gray')
    plt.title('geometricMeanFilter denoised', fontproperties='SimHei')
    plt.subplot(337); plt.imshow(im6, cmap='gray')
    plt.title('medianFiltering denoised', fontproperties='SimHei')
    plt.show()

    im7 = minFilter(im2,3,3)
    im8 = harmonicMeanFilter(im2,3,3)
    im9 = contraharmonicMeanFilter(im2,3,3,.5)
    im10 = contraharmonicMeanFilter(im2,3,3,-.5)

    plt.subplot(231); plt.imshow(im, cmap='gray')
    plt.title('origin', fontproperties='SimHei')
    plt.subplot(232); plt.imshow(im3, cmap='gray')
    plt.title('image with salt noise', fontproperties='SimHei')
    plt.subplot(233); plt.imshow(im7, cmap='gray')
    plt.title('minFilter denoised', fontproperties='SimHei')
    plt.subplot(234)
    plt.imshow(im8, cmap='gray')
    plt.title('harmonicMeanFilter denoised', fontproperties='SimHei')
    plt.subplot(235);
    plt.imshow(im9, cmap='gray')
    plt.title('contraharmonicMeanFilter q = 0.5', fontproperties='SimHei')
    plt.subplot(236);
    plt.imshow(im10, cmap='gray')
    plt.title('contraharmonicMeanFilter denoised q = -0.5', fontproperties='SimHei')

    plt.show()

def task3():
    filename = os.path.join('.', 'hw4_input','task_3','97.png')
    im = imread(filename).astype('float')
    [height,weight,chanel] = np.shape(im)
    newIm = np.zeros([height,weight,chanel])
    rChanel = im[:,:,0]
    gChanel = im[:,:,1]
    bChanel = im[:,:,2]

    rIm = histogramEqualized(rChanel)
    gIm = histogramEqualized(gChanel)
    bIm = histogramEqualized(bChanel)
    newIm[:,:,0],newIm[:,:,1],newIm[:,:,2] = \
        rIm,gIm,bIm

    plt.subplot(331)
    plt.imshow(im.astype(np.uint8))
    plt.subplot(332)
    plt.imshow(newIm.astype(np.uint8))
    plt.subplot(333)
    #newIm = np.ndarray.astype
    newIm = newIm.astype(np.uint8)
    plt.imshow(newIm)

    hist1 = getHistogram(rIm)
    hist2 = getHistogram(gIm)
    hist3 = getHistogram(bIm)

    hist = (hist1 + hist2 + hist3) / 3.
    rIm = histogramMatching(rIm,hist)
    gIm = histogramMatching(gIm,hist)
    bIm = histogramMatching(bIm,hist)

    newIm[:, :, 0], newIm[:, :, 1], newIm[:, :, 2] = \
        rIm, gIm, bIm
    plt.subplot(334)
    plt.imshow(newIm.astype(np.uint8))

    hsiIm = rgb2hsi(im)
    hsiIm[:,:,2] = histogramEqualized(hsiIm[:,:,2])
    rgbIm = hsi2rgb(hsiIm[:,:,0],hsiIm[:,:,1],hsiIm[:,:,2])

    plt.subplot(335)
    plt.imshow(hsiIm[:,:,0], cmap='gray')
    plt.subplot(336)
    plt.imshow(hsiIm[:,:,1], cmap='gray')
    plt.subplot(337)
    plt.imshow(hsiIm[:,:,2], cmap='gray')
    plt.subplot(338)
    plt.imshow(rgbIm)
    print rgbIm
    '''
    from skimage import color
    from temp import *

    plt.subplot(339)
    plt.imshow(newIm[:,:,0].astype(np.uint8), cmap='gray')
    print np.max(newIm),np.min(newIm)

    print newIm[:,:,0]-  hsiIm[:,:,0],'11'
    print newIm[:,:,1]-  hsiIm[:,:,1],'22'
    print newIm[:,:,2]-  hsiIm[:,:,2]  ,'33'
    print newIm[:,:,0]
    '''
    plt.show()




def hw4():
    #task1()
    task2()
    #task3()

if __name__ == '__main__':
    hw4()