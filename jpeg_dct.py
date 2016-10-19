import numpy as np
import re
from PIL import Image

im = []

for i in range(16):
    lin = []
    p = raw_input()
    p = re.split(' *',p)
    for j in range(len(p)):
   	lin.append(int(p[i],16))
    im.append(lin)
im = Image.fromarray(np.array(im,np.uint8),'RGB')
im1 = im.convert("YCbCr")


def _ycc(r, g, b): # in (0,255) range
    y = .299*r + .587*g + .114*b
    cb = 128 -.168736*r -.331364*g + .5*b
    cr = 128 +.5*r - .418688*g - .081312*b
    return y, cb, cr

def _rgb(y, cb, cr):
    r = y + 1.402 * (cr-128)
    g = y - .34414 * (cb-128) -  .71414 * (cr-128)
    b = y + 1.772 * (cb-128)
    return r, g, b


