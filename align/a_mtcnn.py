#%%
from detector import detect_faces
from PIL import Image
import cv2
import numpy as np 
import math
import matplotlib.image as mpimg
from time import time
import functools
def get_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        st = time()
        a = func(*args, **kwargs)
        if __name__ == '__main__':
            print(func.__name__, "time:", time()-st)
        return a
    return wrapper

@get_time
def get_landmarks(img): #in rgb
    floc, fslands = detect_faces(Image.fromarray(img), min_face_size=100)
    face_locations = []
    faces_landmarks = []
    for (x1, y1, x2, y2, p) in floc:
        face_locations.append(list(map(lambda x:int(x), (y1, x2, y2, x1))))
    for flands in fslands:
        face_landmarks = []
        for x in range(5):
            face_landmarks.append((int(flands[x]), int(flands[x+5])))
        faces_landmarks.append(face_landmarks)
    return face_locations, faces_landmarks

def calc_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

def calc_theta(x, y):
    return -math.degrees(math.asin(y / math.sqrt(x*x + y*y)))

glass = cv2.imread("/Users/candy/code/cv/facial_filter/align/glass.png")[:,:,::-1]
glass_all = cv2.imread("/Users/candy/code/cv/facial_filter/align/glass_all.png")[:,:,::-1]
#@get_time
def wear_glass(img, p, glass):
    if p == []:
        return img
    h0, w0, d = glass.shape
    yjj = int(p[1][0] - p[0][0])
    yjj = int(calc_dist(p[1], p[0]))
    w = int(yjj * 3)
    h = int((w/w0) * h0)
    nh = h + (w-h)//2 * 2
    glass = cv2.resize(glass, (w, h), interpolation = cv2.INTER_CUBIC)
    padding = np.zeros(((w-h)//2, w, 3))
    #print(padding.shape, glass.shape)
    glass = np.concatenate((padding, glass, padding), axis=0)
    #print("glass", glass.shape)

    theta = calc_theta(p[1][0]-p[0][0], p[1][1]-p[0][1])
    M = cv2.getRotationMatrix2D((w/2, nh/2), theta, 1)
    glass = cv2.warpAffine(glass, M, (w, nh))
    r1, r2 = max(0, p[1][1]-nh//2), min(img.shape[0], p[1][1]-nh//2+nh)
    c1, c2 = max(0, p[0][0]-yjj), min(img.shape[1], p[0][0]-yjj+w)
    glass = glass[:r2-r1, :c2-c1]
    transparent = glass[:,:,:] != 0
    img[r1:r2, c1:c2][transparent] = glass[transparent]
    return img
@get_time
def plot(img, face_locations, faces_landmarks):
    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmark in faces_landmarks:
            for i, (x, y) in enumerate(landmark):
                font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(img, str(i),(x, y), font, 4, (255,255,255),2,cv2.LINE_AA)
                cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
        img = wear_glass(img.copy(), face_landmarks, glass)
    return img
@get_time
def plot_landmarks(img): #in rgb
    face_locations, faces_landmarks = get_landmarks(img)
    return plot(img, face_locations, faces_landmarks)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img4 = plt.imread("/Users/candy/code/cv/facial_filter/align/twh3.jpg")
    plt.imshow(plot_landmarks(img4))
    plt.show()

    img = plt.imread("twh.jpg")
    plt.imshow(plot_landmarks(img))
    plt.show()
    img2 = plt.imread("a.jpg")
    img3 = plt.imread("twh2.jpg")
    plt.imshow(plot_landmarks(img2))
    plt.show()
    plt.imshow(plot_landmarks(img3))
    plt.show()
