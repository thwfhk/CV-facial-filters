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

img = plt.imread("twh.jpg")
DNN = "TF"
if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector_uint8.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    
conf_threshold = 0.5
@get_time
def plot_landmarks(img): #in rgb
    blob = cv2.dnn.blobFromImage(img, 1.0, img.shape[:2], np.mean(img, axis=2), False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    face_locations = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            face_locations.append((y2, x2, y1, x1))
    
    faces_landmarks = fr.face_landmarks(img, face_locations, model='large')

    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmarks in faces_landmarks:
            for landmark in landmarks.values():
                for (x, y) in landmark:
                    cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
    return img

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def plot_landmarks(img): #in rgb
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_locations = face_cascade.detectMultiScale(gray_img, 
        scaleFactor=1.05,
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

plt.imshow(plot_landmarks(img))
plt.show()

