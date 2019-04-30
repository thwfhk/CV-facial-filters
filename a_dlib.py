#%%
import face_recognition as fr 
import dlib
from imutils import face_utils
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

hogFaceDetector = dlib.get_frontal_face_detector()
predictor68 = dlib.shape_predictor("dlib_data/shape_predictor_68_face_landmarks.dat")
predictor5 = dlib.shape_predictor("dlib_data/shape_predictor_5_face_landmarks.dat")
    
@get_time
def get_landmarks(img, predictor): #in rgb
    faceRects = hogFaceDetector(img, 0)
    face_locations = []
    faces_landmarks = []
    for faceRect in faceRects:
        face_locations.append((faceRect.top(), faceRect.right(), faceRect.bottom(), faceRect.left()))
        shape = predictor(img, faceRect)
        face_landmarks = face_utils.shape_to_np(shape)
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
def plot_landmarks(img, predictor): #in rgb
    floc, fland = get_landmarks(img, predictor)
    return plot(img, floc, fland)

if __name__ == '__main__':
    img = plt.imread("a.jpg")
    plt.imshow(plot_landmarks(img, predictor5))
    plt.show()
