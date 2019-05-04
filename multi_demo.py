import face_recognition as fr
import cv2
from multiprocessing import Process, Manager, cpu_count
import time
import numpy as np
#import matplotlib.pyplot as plt
import a_dlib

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
        face_locations, faces_landmarks = a_dlib.get_landmarks(rgb_frame, a_dlib.predictor5)
        frame_process = a_dlib.plot(rgb_frame, face_locations, faces_landmarks)[:,:,::-1]

        # Wait to write
        while Global.write_num != worker_id:
            time.sleep(nap)
        # Send frame to global
        write_frame_list[worker_id] = frame_process
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
    worker_num = 4

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
            elif fps < 20:
                Global.frame_delay = (1 / fps) * 0.5
            elif fps < 30:
                Global.frame_delay = (1 / fps) * 0.25
            else:
                Global.frame_delay = 0

            # Display the resulting image
            cv2.imshow('Video', write_frame_list[prev_id(Global.write_num)])

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Global.is_exit = True
            break

        time.sleep(nap)

    # Quit
    cv2.destroyAllWindows()