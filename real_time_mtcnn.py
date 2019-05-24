# import face_recognition as fr
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


import a_mtcnn


# 使用rgb，进行了镜面处理
def addFilters(frame, selected_filters):
    frame = frame[:,::-1,:]
    awsl = a_mtcnn.get_landmarks(frame)
    return a_mtcnn.plot(frame, *awsl, selected_filters)

selected_filters = {"eye":"glass", "ear":"rabbit_ear", "nose":"cat_nose"}

if __name__ == '__main__':
    process_prob = 1
    process_cnt = 0

    #video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture('1.mp4')
    print(video_capture.get(5))
    cv2.namedWindow('meow')

    w, h = int(video_capture.get(3)), int(video_capture.get(4))
    x = (w-h)//2
    mask = np.zeros((h, w, 3), dtype='bool')
    mask[:, x:w-x, :] = True


    fps_list = []
    tmp_time = time.time()
    while True:
        process_cnt += 1
        ret, frame = video_capture.read()
        frame = frame[:,::-1,:]
        frame = frame[mask].reshape(h, h, 3)
        rgb_frame = frame[:,:,::-1].copy()
        if process_cnt % process_prob == 0:
            awsl = a_mtcnn.get_landmarks(rgb_frame)
        frame = a_mtcnn.plot(rgb_frame, *awsl, selected_filters)[:,:,::-1]
        cv2.imshow('meow', frame)

        delay = time.time() - tmp_time
        tmp_time = time.time()
        fps_list.append(delay)
        if len(fps_list) > 5:
            fps_list.pop(0)
        fps = len(fps_list) / np.sum(fps_list)
        print("fps: %.2f" % fps)

        time3 = time.time()
        time.sleep(0.01)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
