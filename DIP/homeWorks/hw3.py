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
    for i in range(1,int(np.log(dim + 0.5)/np.log(2))):
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

def testDft():
    filename = os.path.join('..', 'hw3_input', '97.png')
    im = imread(filename)
    oim = im
    [height, weight] = np.shape(im)
    plt.subplot(221)
    plt.imshow(im, 'gray')
    newIm = getSpectrumIm(im)
    plt.subplot(222)
    plt.imshow(newIm.astype(np.uint8), 'gray')
    plt.show()

def hw3():
    testDft()

if __name__ == '__main__':
    hw3()