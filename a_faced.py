#%%
import face_recognition as fr 
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

from faced import FaceDetector
from faced.utils import annotate_image
face_detector = FaceDetector()

img = plt.imread("twh.jpg")
img2 = plt.imread("a.jpg")
@get_time
def plot_landmarks(img): #in rgb
    thresh = 0.85
    face_boxes = face_detector.predict(img, thresh)
    face_locations = []
    for (x, y, w, h, prob) in face_boxes:
        face_locations.append(list(map(lambda x:int(x), (y-h/2, x+w/2, y+h/2, x-w/2))))
    faces_landmarks = fr.face_landmarks(img, face_locations, model='large')

    print(face_locations)
    #return img
    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmarks in faces_landmarks:
            for landmark in landmarks.values():
                for (x, y) in landmark:
                    cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
    return img
# Receives RGB numpy image (HxWxC) and
# returns (x_center, y_center, width, height, prob) tuples. 
# Use this utils function to annotate the image.
#ann_img = annotate_image(img, bboxes)

plot_landmarks(img)
plot_landmarks(img2)
#plt.imshow(plot_landmarks(img))
#plt.imshow(plot_landmarks(img2))
#plt.show()



#%%
