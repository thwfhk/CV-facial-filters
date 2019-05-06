import face_recognition as fr
import cv2
from multiprocessing import Process, Manager, cpu_count
import time
import numpy as np
#import matplotlib.pyplot as plt
import a_mtcnn

nap = 0.01
# Get next worker's id
def next_id(current_id):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1

# Get previous worker's id
def prev_id(current_id):
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1


# resize the img from 720*1280 to 720*720
mask = np.zeros((720, 1280, 3), dtype='bool')
mask[:, 280:1000, :] = True

# A subprocess use to capture frames.
def capture(read_frame_list):
    video_capture = cv2.VideoCapture(0)
    # video_capture.set(5, 30) # Frame rate.
    print("Width: %d, Height: %d, FPS: %d" % (720, 720, video_capture.get(5)))

    while not Global.is_exit:
        # If it's time to read a frame
        if Global.buff_num != next_id(Global.read_num):
            # Grab a single frame of video
            ret, frame = video_capture.read()
            frame = frame[:,::-1,:] #mirror
            frame = frame[mask].reshape(720, 720, 3) #resize
            read_frame_list[Global.buff_num] = frame
            Global.buff_num = next_id(Global.buff_num)
        else:
            time.sleep(nap)

    # Release webcam
    video_capture.release()


def calc(x, y, th1=20, th2=50):
    if abs(x-y) < th1:
        return y
    elif abs(x-y) < th2:
        return int((x+y)/2)
    else:
        return x
def correction(loc1, land1, loc2, land2):
    loc  = []
    land = []
    for (t1, r1, b1, l1), (t2, r2, b2, l2) in zip(loc1, loc2):
        t, r, b, l = calc(t1, t2), calc(r1, r2), calc(b1, b2), calc(l1, l2)
        loc.append((t, r, b, l))
    for i1, i2 in zip(land1, land2):
        i = []
        for (x1, y1), (x2, y2) in zip(i1, i2):
            x, y = calc(x1, x2, 10, 20), calc(y1, y2, 10, 20)
            i.append((x, y))
        land.append(i)
    return loc, land

    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmark in faces_landmarks:
            for (x, y) in landmark:
                cv2.circle(img, (x, y), 2, (255, 255, 255), 2)

# Many subprocess use to process frames.
def process(worker_id, read_frame_list, write_frame_list):
    while not Global.is_exit:
        # Wait to read
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num):
            time.sleep(nap)

        # Delay to make the video look smoother
        time.sleep(Global.frame_delay)
        # Read a single frame from frame list
        frame_process = read_frame_list[worker_id]
        # Expect next worker to read frame
        Global.read_num = next_id(Global.read_num)

        rgb_frame = frame_process[:, :, ::-1].copy()
        face_locations, faces_landmarks = a_mtcnn.get_landmarks(rgb_frame)

        # Wait to write
        while Global.write_num != worker_id:
            time.sleep(nap)
        if prev_id(worker_id) in write_frame_list:
            (_img, _loc, _land) = write_frame_list[prev_id(worker_id)]
            if _loc != [] and _land != []:
                face_locations, faces_landmarks = correction(face_locations, faces_landmarks, _loc, _land)
        frame_process = a_mtcnn.plot(rgb_frame, face_locations, faces_landmarks)[:,:,::-1]
        # Send frame to global
        write_frame_list[worker_id] = (frame_process, face_locations, faces_landmarks)
        # Expect next worker to write frame
        Global.write_num = next_id(Global.write_num)


if __name__ == '__main__':

    # Global variables
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    # Number of workers (subprocess use to process frames)
    #worker_num = cpu_count()
    worker_num = 3

    # Subprocess list
    p = []

    # Create a subprocess to capture frames
    p.append(Process(target=capture, args=(read_frame_list,)))
    p[0].start()

    # Create workers
    for worker_id in range(1, worker_num + 1):
        p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list)))
        p[worker_id].start()

    # Start to show video
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)

            # Calculate fps
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(delay)
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(fps_list) / np.sum(fps_list)
            print("fps: %.2f" % fps)

            # Calculate frame delay, in order to make the video look smoother.
            # When fps is higher, should use a smaller ratio, or fps will be limited in a lower value.
            # Larger ratio can make the video look smoother, but fps will hard to become higher.
            # Smaller ratio can make fps higher, but the video looks not too smoother.
            # The ratios below are tested many times.
            if fps < 30:
                Global.frame_delay = 0

            # Display the resulting image
            cv2.imshow('Video', write_frame_list[prev_id(Global.write_num)][0])

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Global.is_exit = True
            time.sleep(0.1)
            break

        time.sleep(nap)

    # Quit
    cv2.destroyAllWindows()