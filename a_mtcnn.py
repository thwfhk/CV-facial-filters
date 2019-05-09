#%%
import mtcnn
import filters
import face_recognition as fr
import dlib
from imutils import face_utils
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

predictor68 = dlib.shape_predictor("dlib_data/shape_predictor_68_face_landmarks.dat")
predictor5 = dlib.shape_predictor("dlib_data/shape_predictor_5_face_landmarks.dat")

@get_time
def get_landmarks(img, predictor = predictor5): #in rgb
    floc, fslands = mtcnn.detect_faces(Image.fromarray(img), min_face_size=100)
    face_locations = []
    faces_landmarks = []
    for (x1, y1, x2, y2, p) in floc:
        face_locations.append(list(map(int, (y1, x2, y2, x1))))
        # dlib
        faceRect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
        shape = predictor(img, faceRect)
        face_landmarks = face_utils.shape_to_np(shape)
        faces_landmarks.append(face_landmarks)

    ''' mtcnn
    for flands in fslands:
        face_landmarks = []
        for x in range(5):
            face_landmarks.append((int(flands[x]), int(flands[x+5])))
        faces_landmarks.append(face_landmarks)
    '''
    # fr
    # faces_landmarks = fr.face_landmarks(img, face_locations, model='large')
    return face_locations, faces_landmarks


@get_time
def plot(img, face_locations, faces_landmarks):
    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        #for landmark in face_landmarks.values():
            #for i, (x, y) in enumerate(landmark):
        for (x, y) in face_landmarks:
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(img, str(i),(x, y), font, 4, (255,255,255),2,cv2.LINE_AA)
            cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
        # img = filters.wear_glass(img.copy(), face_landmarks, "glass")
    return img
@get_time
def plot_landmarks(img): #in rgb
    face_locations, faces_landmarks = get_landmarks(img)
    return plot(img, face_locations, faces_landmarks)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img1 = plt.imread("twh.jpg")
    plt.imshow(plot_landmarks(img1))
    plt.show()

    img2 = plt.imread("a.jpg")
    plt.imshow(plot_landmarks(img2))
    plt.show()
