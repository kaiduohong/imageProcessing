#-*- coding:utf8 -*-
#author:hong kaiduo
import numpy as np
import math
import re
import sys
from matplotlib import pyplot as plt
import os
import matplotlib as mpl
import math


'''
def rgb2yCbCr(im):
    row, col = np.shape(im)
    ycbcrImg = np.zeros(row * col * 3).reshape(row, col, 3)
    for i in range(row):
        for j in range(col):
            ycbcrImg[i, j] = _ycc(im[i, j])
    return ycbcrImg
def yCbCr2rgb(im):
    row, col = np.shape(im)
    rgbImg = np.zeros(row * col * 3).reshape(row, col, 3)
    for i in range(row):
        for j in range(col):
            ycbcrImg[i, j] = _rgb(im[i, j])
    return rgbImg
'''

#dct变换

def jpegEncode(f):
    f = np.array(f,dtype='float')
    row, col = np.shape(f)
    if row % 8 != 0:
        r = 8 - row % 8
        #将行的最后一行复制，凑成８的倍数
        f = np.append(f,np.repeat([f[-1]],r,0),0)
    #对列做同样的操作
    if col % 8 != 0:
        c = 8 - col % 8
        f = np.append(f,np.repeat([f[:,-1]],c,0).transpose(),1)
    row,col = np.shape(f)

    F = np.zeros(row * col).reshape(row,col)
    #dct
    for i in range(row / 8):
        for j in range(col  / 8):
            print 'how many'
            rowH = i * 8
            rowT = rowH + 8
            colH = i * 8
            colT = colH + 8
            F[rowH:rowT,colH:colT] = \
                _twoDimFdctMatrixForm(f[rowH:rowT,colH:colT])

    return F
#跟encode差不多
def jpegDecode(F):
    F = np.array(F, dtype='float')
    #print np.shape(F)
    row, col = np.shape(F)
    if row % 8 != 0:
        r = 8 - row % 8
        # 将行的最后一行复制，凑成８的倍数
        F = np.append(F, np.repeat([F[-1]], r, 0), 0)
    # 对列做同样的操作
    if col % 8 != 0:
        c = 8 - col % 8
        F = np.append(F, np.repeat([F[:, -1]], c, 0).transpose(), 1)
    row, col = np.shape(F)

    f = np.zeros(row * col).reshape(row, col)
    #idct
    for i in range(row / 8):
        for j in range(col / 8):
            rowH = i * 8
            rowT = rowH + 8
            colH = i * 8
            colT = colH + 8
            f[rowH:rowT, colH:colT] = \
                _twoDimIdctMatrixForm(F[rowH:rowT, colH:colT])

    return f

#返回矩阵形式
def _genFdctEff(num):
    E = np.zeros(num*num).reshape(num, num)
    c = np.zeros(num) + np.sqrt(2.0 / num);
    c[0] = np.sqrt(1.0 / num)
    print c,np.sqrt(2.0/num)
    for i in range(num):
        for j in range(num):
            E[i,j] = c[i] * np.cos((j + 0.5) * np.pi * i / num)

    return np.matrix(E)
#一维idct
#用矩阵形式的
def _twoDimFdctMatrixForm(f):
    row,col = np.shape(f)

    F = np.zeros(row * col).reshape(row,col)

    #表达成矩阵的形式,F = Cleft * f * Cright
    Cleft = _genFdctEff(row)
    Cright = _genFdctEff(col).transpose()

    F = Cleft * np.matrix(f) * Cright

    return np.array(F)

#一维dct
def _dct(f):
    N = np.shape(f)
    F = np.zeros(N)
    for i in range(N):
        if i == 0:
            c = np.sqrt(1.0 / N)
        else:
            c = np.sqrt(2.0 / N)
        sums = 0
        for j in range(N):
            sums += f[j] * np.cos(np.PI * (j + 0.5) * i / N)
        F[i] = c * sums
    return F

#用拆分形式的
def _twoDimDct(f):
    row, col = np.shape(f)
    F = np.zeros(row, col)

    for i in range(row):
        F[i] = _dct(f[i])
    for j in range(col):
        F[:,j] = _dct(F[:,j])

    return F

def _genIdctEff(num):
    E = np.zeros(num * num).reshape(num, num)
    c = np.zeros(num) + np.sqrt(2.0 / num);
    c[0] = np.sqrt(1.0 / num)

    for i in range(num):
        for j in range(num):
            E[i, j] = c[j] * np.cos((i + 0.5) * np.pi * j / num)
    return np.matrix(E)
#矩阵形式
def _twoDimIdctMatrixForm(F):
    row,col = np.shape(F)

    f = np.zeros(row * col).reshape(row,col)

    #表达成矩阵的形式,f = Cleft * F * Cright
    Cleft = _genIdctEff(row)
    Cright = _genIdctEff(col).transpose()

    f = Cleft * np.matrix(F) * Cright

    return np.array(f)
#拆分形式
def twoDimIdct(F):
    row, col = np.shape(F)
    f = np.zeros(row, col)

    for i in range(row):
        f[i] = _idct(F[i])
    for j in range(col):
        f[:, j] = _idct(f[:, j])

    return f
#一维形式
def _idct(F):
    N = np.shape(F)
    f = np.zeros(N)

    for i in range(N):
        sums = 0
        for j in range(N):
            if j == 0:
                c = np.sqrt(1.0 / N)
            else:
                c = np.sqrt(2.0 / N)
            sums += c * F[j] * np.cos(np.PI * (i + 0.5) * j / N)
        f[i] = sums
    return f

def quantization(im):
    #读取亮度量化表
    file = open('./jpegEncode/data/quantizationTable.txt')
    quantizationTable = []
    for line in file:
        lin = []
        p = re.split(' *',line)
        for j in range(len(p)):
            lin.append(p[j])
        quantizationTable.append(lin)
    quantizationTable = np.array(quantizationTable)


'''
def _ycc(r, *arg): # in (0,255) range
    #如果是灰度图，那么输入就只有一个灰度，否则就是rgb
    if len(arg)!= 0 and len(arg) != 2:
        raise Exception 'the number of argument is wrong'
    if len(arg) == 0:
        g = b = r
    else:
        g = arg[0]
        b = arg[1]

    y = .299*r + .587*g + .114*b
    cb = 128 -.168736*r -.331364*g + .5*b
    cr = 128 +.5*r - .418688*g - .081312*b
    return y, cb, cr

def _rgb(y, cb, cr):
    r = y + 1.402 * (cr-128)
    g = y - .34414 * (cb-128) -  .71414 * (cr-128)
    b = y + 1.772 * (cb-128)
    return r, g, b

'''

if __name__ == '__main__':
    im = []
    #读取lena数据（文本），存储到im中
    lenafile = open('./jpegEncode/data/test.dat','r')

    for line in lenafile:
        lin = []
        p = re.split(' *',line)
        for j in range(len(p)):
       	    lin.append(int(p[j]))
        im.append(lin)

    lenafile.close()

    print im












    #先存成float　方便运算
    im = np.array(im,dtype='float')
    [row, col] = np.shape(im)
    temp = np.ndarray.astype(im,'uint8')

    plt.imshow(temp,mpl.cm.gray_r)


    print 'row = ',row,' col = ',col

    '''#RGB转成YCbCr
    #ycbcrImg = rgb2yCbCr(im)
    #F1 = jpegEncode(ycbcrImg[:,:,0])
    #F2 = jpegEncode(ycbnp.sqrt(crImg[:,:,1])
    #F3 = jpegEncode(ycbcrImg[:,:,2])
    #ycbcrImg = jpegDecode(F)
    '''
    fimg = jpegEncode(im - 128)

    fimg = quantization(fimg)


    im = np.array(jpegDecode(fimg) + 128,'uint8')

    plt.imshow(im,mpl.cm.gray_r)
    plt.show()
    raw_input('pause_pause')
