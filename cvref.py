import numpy as np
from cv2 import cv2

def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, a, s):
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
