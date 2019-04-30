import face_recognition as fr
import cv2
import numpy as np
import dlib
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

video_capture = cv2.VideoCapture(0)

@get_time
def plot_landmarks(img): #in rgb
    print(img.shape)
    face_locations = fr.face_locations(img, model='hog')
    faces_landmarks = fr.face_landmarks(img, face_locations, model='large')

    print(face_locations)
    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmarks in faces_landmarks:
            for landmark in landmarks.values():
                for (x, y) in landmark:
                    cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
    return img

@get_time
def plot_landmarks_opencv(img): #in rgb
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_locations = face_cascade.detectMultiScale(gray_img, 
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print(face_locations)
    li = []
    for (x, y, w, h) in face_locations:
        top, bottom, left, right = y, y+h, x, x+w
        li.append((top, right, bottom, left))
    face_locations = li
    faces_landmarks = fr.face_landmarks(img, face_locations, model='large')

    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmarks in faces_landmarks:
            for landmark in landmarks.values():
                for (x, y) in landmark:
                    cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
    return img

hogFaceDetector = dlib.get_frontal_face_detector()
@get_time
def plot_landmarks_dlib_hog(img): #in rgb
    faceRects = hogFaceDetector(img, 0)
    face_locations = []
    for faceRect in faceRects:
        face_locations.append((faceRect.top(), faceRect.right(), faceRect.bottom(), faceRect.left()))
    faces_landmarks = fr.face_landmarks(img, face_locations, model='large')

    print(face_locations)
    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmarks in faces_landmarks:
            for landmark in landmarks.values():
                for (x, y) in landmark:
                    cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
    return img

from faced import FaceDetector
from faced.utils import annotate_image
face_detector = FaceDetector()


@get_time
def plot_landmarks_faced(img): #in rgb
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

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:,:,::-1].copy()
    frame = plot_landmarks_faced(rgb_frame)[:,:,::-1]
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()