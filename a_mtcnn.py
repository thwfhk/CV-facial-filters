#%%
import mtcnn
from MEOW3DDFA import a_3ddfa
import filters
# import face_recognition as fr
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

import kalman_3d
kalman_list = []
for i in range(68):
    kalman_list.append(kalman_3d.MyKalman())

def use_kalman(pts_3d):
    new_pts = []
    for i in range(3):
        new_pts.append([0 for j in range(68)])
    for i in range(68):
        x, y, z = pts_3d[0][i], pts_3d[1][i], pts_3d[2][i]
        cur = kalman_list[i]
        x, y, z = cur.update(x, y, z)
        new_pts[0][i], new_pts[1][i], new_pts[2][i] = x, y, z
    print(np.array(pts_3d) - np.array(new_pts))
    return new_pts.copy()


@get_time
def get_landmarks(img, predictor = predictor5): #in rgb
    floc, fslands = mtcnn.detect_faces(Image.fromarray(img), min_face_size=100)
    face_locations = []
    faces_landmarks = []
    faceRects = []
    for (x1, y1, x2, y2, p) in floc:
        face_locations.append(list(map(int, (y1, x2, y2, x1))))
        faceRects.append(dlib.rectangle(int(x1), int(y1), int(x2), int(y2)))
        ''' dlib
        faceRect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
        shape = predictor(img, faceRect)
        face_landmarks = face_utils.shape_to_np(shape)
        faces_landmarks.append(face_landmarks)
        '''

    # NOTE: using 3ddfa
    faces_landmarks, Ps, poses, pts_3ds, roi_boxes = a_3ddfa.meow_landmarks(img, faceRects)
    '''
    if pts_3ds != []:
        print("use")
        pts_3ds[0] = use_kalman(pts_3ds[0].copy())
    '''
    
    return face_locations, faces_landmarks, Ps, poses, pts_3ds, roi_boxes


@get_time
def plot(img, face_locations, faces_landmarks, Ps, poses, pts_3ds, roi_boxes, selected_filters):
    '''
    for loc, points, P, pose, pts_3d, roi_box in zip(face_locations, faces_landmarks, Ps, poses, pts_3ds, roi_boxes):
        (top, right, bottom, left) = loc
        #cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        for i in range(points.shape[1]):
            x, y, z = points[0, i], points[1, i], points[2, i]
            #cv2.circle(img, (x, y), 1, (255, 255, 255), 2)
            # cv2.putText(img, str(i),(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1,cv2.LINE_AA)
        #img = filters.qwq(img.copy(), points, P, pose, pts_3d, roi_box)
        #img = filters.wear_glass(img.copy(), points, P, pose, pts_3d, roi_box)
        #img = filters.wear_ears(img.copy(), points, P, pose, pts_3d, roi_box)
        #img = filters.wear_nose(img.copy(), points, P, pose, pts_3d, roi_box)
    '''
    for P, pts_3d, roi_box in zip(Ps, pts_3ds, roi_boxes):
        img = filters.add_filters(img, P, pts_3d, roi_box, selected_filters)
    return img

selected_filters = {"eye":"glass", "ear":"rabbit_ear", "nose":"cat_nose"}

@get_time
def plot_landmarks(img, selected_filters): #in rgb
    face_locations, faces_landmarks, Ps, poses, pts_3ds, roi_boxes = get_landmarks(img)
    return plot(img, face_locations, faces_landmarks, Ps, poses, pts_3ds, roi_boxes, selected_filters)

# 使用rgb，进行了镜面处理
def addFilters(frame, selected_filters):
    frame = frame[:,::-1,:]
    awsl = get_landmarks(frame)
    return plot(frame, *awsl, selected_filters)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img_name_list = ["ts1.jpg", "s.jpg", "t.jpg"]
    for name in img_name_list:
        img = plt.imread(name)
        res = plot_landmarks(img, selected_filters)
        plt.imshow(res)
        plt.show()