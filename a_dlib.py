#%%
import dlib
from imutils import face_utils
import cv2
import numpy as np 
import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
import time
import functools
def get_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        st = time.time()
        a = func(*args, **kwargs)
        if __name__ == "__main__":
            print(func.__name__, "time:", time.time()-st)
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
        for (x, y) in face_landmarks:
            cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
    return img
@get_time
def plot_landmarks(img, predictor = predictor68): #in rgb
    floc, fland = get_landmarks(img, predictor)
    return plot(img, floc, fland)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    glass = cv2.imread("glass")
    img = plt.imread("a.jpg")
    img2 = plt.imread("twh.jpg")
    img3 = plt.imread("twh2.jpg")
    plt.imshow(plot_landmarks(img))
    plt.show()
    plt.imshow(plot_landmarks(img2))
    plt.show()
    plt.imshow(plot_landmarks(img3))
    plt.show()