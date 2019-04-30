#%%
from detector import detect_faces
from PIL import Image
import cv2
import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from time import time
import functools
def get_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        st = time()
        a = func(*args, **kwargs)
        print(func.__name__, "time:", time()-st)
        return a
    return wrapper

@get_time
def get_landmarks(img): #in rgb
    floc, fslands = detect_faces(Image.fromarray(img))
    face_locations = []
    faces_landmarks = []
    for (x1, y1, x2, y2, p) in floc:
        face_locations.append(map(lambda x:int(x), (y1, x2, y2, x1)))
    for flands in fslands:
        face_landmarks = []
        for x in range(5):
            face_landmarks.append((flands[x], flands[x+5]))
        faces_landmarks.append(face_landmarks)
    return face_locations, faces_landmarks

@get_time
def plot(img, face_locations, faces_landmarks):
    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmark in faces_landmarks:
            for (x, y) in landmark:
                cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
    return img
@get_time
def plot_landmarks(img): #in rgb
    face_locations, faces_landmarks = get_landmarks(img)
    return plot(img, face_locations, faces_landmarks)

if __name__ == '__main__':
    img = plt.imread("a.jpg")
    img2 = plt.imread("twh.jpg")
    img3 = plt.imread("twh2.jpg")
    plt.imshow(plot_landmarks(img))
    plt.imshow(plot_landmarks(img2))
    plt.imshow(plot_landmarks(img3))
    plt.show()
