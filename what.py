import cv2
import numpy as np
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


video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture('a.mp4')
video_capture.set(5, 25)
time.sleep(1)
mask = np.zeros((720, 1280, 3), dtype='bool')
mask[:, 280:1000, :] = True

fps_list = []
tmp_time = time.time()
while True:
    time1 = time.time()
    ret, frame = video_capture.read()
    #frame = frame[:,::-1,:]
    #frame = frame[mask].reshape(720, 720, 3)
    print("time1 {}".format(time.time()-time1))
    time2 = time.time()
    
    cv2.imshow('Video', frame)
    print("time2 {}".format(time.time()-time2))

    delay = time.time() - tmp_time
    tmp_time = time.time()
    fps_list.append(delay)
    if len(fps_list) > 5:
        fps_list.pop(0)
    fps = len(fps_list) / np.sum(fps_list)
    print("fps: %.2f" % fps)

    time3 = time.time()
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #if 0xFF == ord('q'):
    #    break
    time.sleep(0.01)
    print("time3 {}".format(time.time()-time3))

video_capture.release()
cv2.destroyAllWindows()
