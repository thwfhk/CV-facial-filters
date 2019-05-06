import filters
from PIL import Image
import cv2
import numpy as np 
import math
import matplotlib.image as mpimg

def calc_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

def calc_theta(x, y):
    return -math.degrees(math.asin(y / math.sqrt(x*x + y*y)))

def wear_glass(img, p, filter_name): #image, points_5, filter_image
    glass = cv2.imread("./filters_image/"+filter_name+".png", cv2.IMREAD_UNCHANGED)
    glass[:, :, [0, 2]] = glass[:, :, [2, 0]] # bgr -> rgb
    if p == []:
        return img
    h0, w0, d = glass.shape
    yjj = int(calc_dist(p[1], p[0]))
    w = int(yjj * 3)
    h = int((w/w0) * h0)
    nh = h + (w-h)//2 * 2 # w>h
    glass = cv2.resize(glass, (w, h), interpolation = cv2.INTER_CUBIC)
    padding = np.zeros(((w-h)//2, w, 4))
    glass = np.concatenate((padding, glass, padding), axis=0)
    #print("glass", glass.shape)

    theta = calc_theta(p[1][0]-p[0][0], p[1][1]-p[0][1])
    M = cv2.getRotationMatrix2D((w/2, nh/2), theta, 1)
    glass = cv2.warpAffine(glass, M, (w, nh))

    r1, r2 = max(0, p[1][1]-nh//2), min(img.shape[0], p[1][1]-nh//2+nh)
    c1, c2 = max(0, p[0][0]-yjj), min(img.shape[1], p[0][0]-yjj+w)
    glass = glass[:r2-r1, :c2-c1]
    transparent = glass[:,:,3] != 0
    glass = glass[:,:,:3] # convert to 3 channels
    img[r1:r2, c1:c2][transparent] = glass[transparent]
    return img

def put_filter(img, p, filter_name):
    filter_image = cv2.imread("./filters_image/"+filter_name+".png")[:,:,::-1]
    d = {}
    with open("./filters_image/glass.txt", "r") as f:
        for line in f:
            ind, x, y = list(map(int, line.split()))
            d[ind] = (x, y)
    

if __name__ == "__main__":
    wear_glass(None, 1, "glass")