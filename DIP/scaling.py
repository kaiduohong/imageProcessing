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
#reload(sys)
#sys.setdefaultencoding('utf-8')



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
    [height,weight] = np.shape(im)
    for i in range(height):
        for j in range(weight):
            im[i,j] = np.round(((olevel - 1) / (level - 1)) * \
                               np.floor(im[i,j] * level / olevel))

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


def dft(v,dim):
    u = np.zeros(dim, dtype=complex)
    for i in range(dim):
        sum = 0j
        for j in range(dim):
            sum += v[j] * np.exp(-2 * np.pi * 1j * i * j/ dim)
        u[i] = sum
    return u

def idft(v,dim):
    u = dft(v, dim) / dim
    v = np.zeros(dim,dtype=complex)
    v[0] = u[0]
    for i in range(dim - 1):
        v[i + 1] = u[dim - i - 1]
    return v

def dft2d(matrix,m,n):
    u = np.zeros([m,n],dtype=complex)
    for i in range(m):
        u[i] = dft(matrix[i],n)
    u = u.transpose()
    for i in range(n):
        u[i] = dft(u[i],m)
    return u.transpose()

def idft2d(matrix,m,n):
    u = np.zeros([m, n], dtype=complex)
    for i in range(m):
        u[i] = idft(matrix[i], n)
    u = u.transpose()
    for i in range(n):
        u[i] = idft(u[i], m)
    return u.transpose()

def getSpectrumIm(im):
    [height, weight] = np.shape(im)
    im = im.astype(np.float)
    for i in range(height):
        for j in range(weight):
            im[i,j] *= (-1)**((i&1)^(j&1))

    im = np.log(np.abs(dft2d(im,height,weight)) + 1)
    return im

def exchange(v, dim):
    j = dim / 2
    for i in range(dim - 1):
        if i < j:
            v[i],v[j] = v[j],v[i]
        k = dim / 2
        while j >= k:
            j -= k
            k /= 2
        if j < k:
            j += k
    return v

def fft(v, dim, f = 1):
    pad = 1
    while pad * 2 <= dim:
        pad *= 2
    u = np.zeros(pad)
    u[:dim] = v
    dim = pad

    v = exchange(u,dim)
    h = 1
    for i in ragne(1,int(np.log(dim + 0.5)/np.log(2))):
        h = h * 2
        wn = np.exp(- f * 2 * np.pi / h)
        w = 1 + 0j
        for k in range(i, i + h / 2):
            x = v[k]
            y = v[k + h / 2]
            v[k] = x + y
            v[k + h / 2] = u - x
            w *= wn
    return v


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


def testDft():
    filename = os.path.join('.', 'hw3_input', '34.png')
    im = imread(filename)
    oim = im
    [height, weight] = np.shape(im)
    plt.subplot(221)
    plt.imshow(im, mpl.cm.gray_r)
    newIm = getSpectrumIm(im)
    print np.max(newIm)
    print np.min(newIm)
    plt.subplot(222)
    plt.imshow(newIm.astype(np.uint8), mpl.cm.gray_r)

    #newIm = dft2d(im,height,weight)
    #iIm = idft2d(newIm,height,weight)
    #plt.subplot(223)
    #plt.imshow(iIm, mpl.cm.gray_r)


    plt.show()

def testFilter():
    filename = os.path.join('.', 'hw3_input', '97.png')
    im = imread(filename)
    [height, weight] = np.shape(im)
    plt.subplot(221)
    plt.imshow(-im, mpl.cm.gray_r)
    filter = np.ones([7,7]) / 49
    newIm = filter2d(im,filter)
    plt.subplot(222)
    plt.imshow(-newIm, mpl.cm.gray_r)

    laplacian = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    newIm = np.abs(filter2d(im, laplacian) )
    plt.subplot(223)
    plt.imshow(-newIm, mpl.cm.gray_r)

    #print newIm
    #laplacian = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #newIm = filter2d(im, laplacian)
    newIm = im + newIm
    plt.subplot(224)
    #print newIm
    plt.imshow(-newIm, mpl.cm.gray_r)

    plt.show()

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

def task1():
    filename = os.path.join('.', 'hw4_input', 'task_1.png')
    im = imread(filename) / 255.
    tbtMeanFilter = np.ones([3, 3]) / 9.
    nbnMeanFilter = np.ones([9, 9]) / 81.
    im1 = filter2d(im, tbtMeanFilter)
    im2 = filter2d(im, nbnMeanFilter)
    im3 = harmonicMeanFilter(im, 3, 3)
    im4 = harmonicMeanFilter(im, 9, 9)
    im5 = contraharmonicMeanFilter(im, 3, 3, 1.5)
    im6 = contraharmonicMeanFilter(im, 9, 9, 1.5)

    plt.subplot(331);io.imshow(im)
    plt.subplot(332);plt.imshow(im1, cmap='gray')
    plt.subplot(333);io.imshow(im2)
    plt.subplot(334);plt.imshow(im3, cmap='gray')
    plt.subplot(335); plt.imshow(im4, cmap='gray')
    plt.subplot(336);plt.imshow(im5, cmap='gray')
    plt.subplot(337);plt.imshow(im6, cmap='gray')

    plt.show()

def generateGaussNoise(m,n,mean,sigma):
    np.random.normal()
    return np.random.normal(mean,sigma,size=[m,n])
def addGaussNoise(im,mean=0,sigma=1):
    [height,weight] = np.shape(im)
    noise = generateGaussNoise(height,weight,mean,sigma)
    im = im + noise / 255.
    im[im > 1] = 1.
    im[im < 0] = 0.
    return im
def addSaltPepperNoise(im,p,q):
    [m,n] = np.shape(im)
    for i in range(m):
        for j in range(n):
            if np.random.uniform() <= p:
                    im[i,j] = 0
            elif np.random.uniform() <= q:
                    im[i,j] = 1
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


def task2():
    filename = os.path.join('.', 'hw4_input', 'task_2.png')
    im = imread(filename)
    im = im[:,:,0] / 255.
    im1 = addGaussNoise(im,0,40)
    im2 = addSaltPepperNoise(im,0.2,0.2)
    plt.subplot(331); plt.imshow(im1, cmap='gray')
    plt.subplot(332); plt.imshow(im2, cmap='gray')

    im3 = arithmeticMeanFilter(im1,3,3)
    im4 = geometricMeanFilter(im1,3,3)
    im5 = medianFiltering(im1,3,3)

    plt.subplot(333);
    plt.imshow(im3, cmap='gray')
    plt.subplot(334);
    plt.imshow(im4, cmap='gray')
    plt.subplot(335);
    plt.imshow(im5, cmap='gray')

    im6 = minFilter(im2,3,3)
    im7 = harmonicMeanFilter(im2,3,3)
    im8 = contraharmonicMeanFilter(im2,3,3,0.5)
    im9 = contraharmonicMeanFilter(im2,3,3,-0.5)

    plt.subplot(336); plt.imshow(im6, cmap='gray')
    plt.subplot(337);plt.imshow(im7, cmap='gray')
    plt.subplot(338);plt.imshow(im8, cmap='gray')
    plt.subplot(339);plt.imshow(im9, cmap='gray')
    #plt.figure()
    #plt.subplot(331);plt.imshow(im8, cmap='gray')
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



def testFilter():
    #task1()
    #task2()
    task3()
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
            sum = sum2 = 0
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




if __name__ == '__main__':

    #filename = os.path.join('.', 'hw3_input', '97.png')
    #im = imread(filename)
    """
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
    """
    #testDft()
    testFilter()
    #testC()
    #testFilter()
