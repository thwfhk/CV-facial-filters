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


#from a_faced import plot_landmarks as plot_landmarks_faced
#from a import plot_landmarks as plot_landmarks_fr_hog
import a_dlib
process_prob = 2
process_cnt = 0
face_locations = []
faces_landmarks = []

video_capture = cv2.VideoCapture(0)
mask = np.zeros((720, 1280, 3), dtype='bool')
mask[:, 280:1000, :] = True
while True:
    process_cnt += 1
    ret, frame = video_capture.read()
    frame = frame[mask].reshape(720, 720, 3)
    rgb_frame = frame[:,:,::-1].copy()
    if process_cnt % process_prob == 0:
        face_locations, faces_landmarks = a_dlib.get_landmarks(rgb_frame, a_dlib.predictor5)
    frame = a_dlib.plot(rgb_frame, face_locations, faces_landmarks)[:,:,::-1]
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
