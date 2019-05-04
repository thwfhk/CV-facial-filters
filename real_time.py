import face_recognition as fr
import cv2
import numpy as np
import dlib
import time
import functools

def get_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        st = time.time()
        a = func(*args, **kwargs)
        print(func.__name__, "time:", time.time()-st)
        return a
    return wrapper


#from a_faced import plot_landmarks as plot_landmarks_faced
#from a import plot_landmarks as plot_landmarks_fr_hog
import a_dlib
#import a_faced
#import a_fr
process_prob = 3
process_cnt = 0
face_locations = []
faces_landmarks = []

video_capture = cv2.VideoCapture(0)
print(video_capture.get(5))
cv2.namedWindow('meow')

mask = np.zeros((720, 1280, 3), dtype='bool')
mask[:, 280:1000, :] = True

fps_list = []
tmp_time = time.time()
while True:
    process_cnt += 1
    ret, frame = video_capture.read()
    frame = frame[:,::-1,:]
    frame = frame[mask].reshape(720, 720, 3)
    rgb_frame = frame[:,:,::-1].copy()
    if process_cnt % process_prob == 0:
        face_locations, faces_landmarks = a_dlib.get_landmarks(rgb_frame, a_dlib.predictor5)
        #face_locations, faces_landmarks = a_fr.get_landmarks(rgb_frame)
    frame = a_dlib.plot(rgb_frame, face_locations, faces_landmarks)[:,:,::-1]
    cv2.imshow('meow', frame)

    delay = time.time() - tmp_time
    tmp_time = time.time()
    fps_list.append(delay)
    if len(fps_list) > 5:
        fps_list.pop(0
    fps = len(fps_list) / np.sum(fps_list)
    print("fps: %.2f" % fps)

    time3 = time.time()
    time.sleep(0.01)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
