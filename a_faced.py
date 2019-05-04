#%%
import face_recognition as fr 
import cv2
import dlib
from imutils import face_utils
import numpy as np 
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

predictor68 = dlib.shape_predictor("dlib_data/shape_predictor_68_face_landmarks.dat")
predictor5 = dlib.shape_predictor("dlib_data/shape_predictor_5_face_landmarks.dat")

from faced import FaceDetector
from faced.utils import annotate_image
face_detector = FaceDetector()

@get_time
def get_landmarks(img): #in rgb
    thresh = 0.85
    face_boxes = face_detector.predict(img, thresh)
    face_locations = []
    for (x, y, w, h, prob) in face_boxes:
        face_locations.append(list(map(lambda x:int(x), (y-h/2, x+w/2, y+h/2, x-w/2))))
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
    img = plt.imread("a.jpg")
    img2 = plt.imread("twh.jpg")
    img3 = plt.imread("twh2.jpg")
    plt.imshow(plot_landmarks(img))
    plt.show()
    plt.imshow(plot_landmarks(img2))
    plt.show()
    plt.imshow(plot_landmarks(img3))
    plt.show()