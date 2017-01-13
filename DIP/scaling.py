#-*-coding:utf8-*-
import numpy as np
from scipy.misc import imread, imsave, imresize
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

def getHistogram(im):
    [height,weight] = np.shape(im)
    histogram = np.zeros(256)
    for i in range(height):
        for j in range(weight):
            histogram[int(im[i,j])] += 1
    return histogram

def histogramEqualized(im):
    level = 256
    [height, weight] = np.shape(im)
    histogram = getHistogram(im)
    p = histogram / height / weight
    maps = np.zeros(level)
    newHistogram = np.zeros(level)
    newIm = np.zeros([height,weight])

    ac = 0
    for i in range(level):
        ac = ac + p[i]
        maps[i] = np.round((level - 1) * ac)
    for i in range(level):
        newHistogram[maps[i]] += histogram[i]
    for i in range(height):
        for j in range(weight):
            newIm[i,j] = maps[int(im[i,j])]

    return newHistogram,newIm

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
    im = im * 1.0
    #im = np.array(im[:50,:50],dtype='float')
    #height = weight = 50

    for i in range(height):
        for j in range(weight):
            im[i,j] *= (-1)**((i&1)^(j&1))

    #print im
    #im = np.abs(dft2d(im,height,weight))
    im = np.abs(np.fft.fft2(im))
#    print im,'\n\n\n\n'dft2d(im,height,weight)
    im = np.log(im)
    print im
    im = 255 - im
    maxn = np.max(im)
    minn = np.min(im)
    im = 255 * (im - minn) / (maxn - minn)
    print maxn,minn
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
        for k in range(j, j + h / 2):
            x = v[k]
            y = v[k + h / 2]
            v[k] = x + y
            v[k + h / 2] = u - t
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
                    posi,posj = i + k - h / 2,j + l - 2 / 2
                    if posi < 0 or posi >= height or\
                        posj < 0 or posj >= weight:
                        continue
                    sum += im[posi,posj] * filter[h - k - 1,w - l - 1]
                newIm[i,j] = sum
    return newIm


def rgb2hsi(im):
    [height,weight,dim] =np.shape(im)
    hue = np.zeros([height,weight])
    saturation =  np.zeros([height,weight])
    idensity = np.zeros([height,weight])

    for i in range(height):
        for j in range(weight):
            [r,g,b] = im[i,j,0:3]
            theta = np.arccos(0.5*((r-g)+(r-b))/\
                              (np.sqrt((r-g)**2 + \
                                (r-b)*(g-b) )))/\
                                (2 * np.pi)
            if b <= g:
                hue[i,j] = theta
            else:
                hue[i,j] = 1 - theta
            saturation[i,j] = 1 - min([r,g,b]) / (r+g+b)
            idensity[i,j] = (r+g+b) / 3.0

    return hue,saturation,idensity



    pass
def hsi2rgb(hue,saturation,idensity):
    [height,weight]  = np.shape(hue)
    im = np.zeros([height, weight,3])


    for i in range(height):
        for j in range(weight):
            h,s,ind = hue[i,j],saturation[i,j],idensity[i,j]
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
            im[i,j,:] = r,g,b
    return im


def testDft():
    filename = os.path.join('.', 'hw3_input', '97.png')
    im = imread(filename)
    oim = im
    [height, weight] = np.shape(im)
    plt.subplot(221)
    plt.imshow(im, mpl.cm.gray_r)
    newIm = getSpectrumIm(im)
    plt.subplot(222)
    plt.imshow(newIm, mpl.cm.gray_r)

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

    print newIm
    #laplacian = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #newIm = filter2d(im, laplacian)
    newIm = im + newIm
    plt.subplot(224)
    print newIm
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
    print h
    h = filter2d(h,meanFilter)
    print h

    im = hsi2rgb(h,s,i)
    plt.subplot(336)
    io.imshow(im)

    plt.show()


def testFilter():
    filename = os.path.join('.', 'hw4_input', 'task_1.png')
    im = imread(filename) / 255.
    print np.shape(im)
    tbtMeanFilter = np.ones([3,3]) / 9.
    nbnMeanFilter = np.ones([9,9]) / 81.
    im1 = filter2d(im,tbtMeanFilter)
    im2 = filter2d(im,nbnMeanFilter)

    plt.subplot(331)
    io.imshow(im2)

    plt.subplot(332)
    plt.imshow(im1,cmap='gray')

    plt.subplot(333)
    io.imshow(im1)

    plt.show()



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

    testDft()"""
    #testFilter()
    #testC()
    testFilter()
