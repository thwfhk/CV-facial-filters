import pcn
import face_recognition as fr
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
    li = pcn.detect_face(img[:,:,::-1])
    print(li)
    return None
    face_locations = []
    for window in li:
        x, y, w, h = window.x, window.y, window.width, window.width 
        angle = window.angle
        print(angle)
        face_locations.append((y+h, x+w, y, x))
    faces_landmarks = fr.face_landmarks(img, face_locations, model='large')
    return face_locations, faces_landmarks

@get_time
def plot(img, face_locations, faces_landmarks):
    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmarks in faces_landmarks:
            for landmark in landmarks.values():
                for (x, y) in landmark:
                    cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
    return img
@get_time
def plot_landmarks(img): #in rgb
    floc, fland = get_landmarks(img)
    return plot(img, floc, fland)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img1 = plt.imread("a.jpg")
    img2 = plt.imread("twh.jpg")
    img3 = plt.imread("twh2.jpg")

    plt.imshow(plot_landmarks(img1))
    plt.show()
    plt.imshow(plot_landmarks(img2))
    plt.show()
    plt.imshow(plot_landmarks(img3))
    plt.show()