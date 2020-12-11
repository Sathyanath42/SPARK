import numpy as np
from cv2 import cv2

def translate(image, x, y): #[1,0,tx],[0,10,ty] : -tx shift left, -ty shift up
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, a, s): #s is the sizing , left as 1
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, a, s)
    rotated =  cv2.warpAffine(image, M, (w, h))
    return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int (w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))

    resized = cv2.resize(image, dim, 
        interpolation = inter)  
    return resized

#cv.flip(image,1) : flips the image , 1 horizontal flip. 0 vertical flip
#cropped = image[30:120 , 240:335] : y axis from 30 to 120, x axis from 240 to 335

def contrast(image,pixel,a):
    if a = 0:
        M = np.ones(image.shape, dtype = "uint8") * pixel
        mod = cv2.add(image, M)
    else:
        M = np.ones(image.shape, dtype = "uint8") * pixel
        mod = cv2.subtract(image, M)
    return mod
        

