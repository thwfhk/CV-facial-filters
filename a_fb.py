#%%
from FaceBoxes import fb
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


class twh:
    def __init__(self, mo='cpu'):
        self.landmarkor = a_3ddfa.my3ddfa(mo)

    @get_time
    def get_landmarks(self, img): #in rgb
        floc = fb.detect(img)
        face_locations = []
        faces_landmarks = []
        faceRects = []

        for (x1, y1, x2, y2, p) in floc:
            face_locations.append(list(map(int, (y1, x2, y2, x1))))
            faceRects.append(dlib.rectangle(int(x1), int(y1), int(x2), int(y2)))
        # NOTE: using 3ddfa
        faces_landmarks, Ps, poses, pts_3ds, roi_boxes = self.landmarkor.meow_landmarks(img, faceRects)

        '''
        if pts_3ds != []:
            #old = pts_3ds[0].copy()
            pts_3ds[0] = use_kalman(pts_3ds[0].copy())
        '''
        
        return face_locations, faces_landmarks, Ps, poses, pts_3ds, roi_boxes
        #return face_locations


    @get_time
    def plot(self, img, face_locations, faces_landmarks, Ps, poses, pts_3ds, roi_boxes, selected_filters, fancy_mode):
        #for loc in face_locations:
            #(top, right, bottom, left) = loc
            #cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            #for i in range(points.shape[1]):
                #x, y, z = points[0, i], points[1, i], points[2, i]
                #cv2.circle(img, (x, y), 1, (255, 255, 255), 2)
                # cv2.putText(img, str(i),(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1,cv2.LINE_AA)
            #img = filters.qwq(img.copy(), points, P, pose, pts_3d, roi_box)
            #img = filters.wear_glass(img.copy(), points, P, pose, pts_3d, roi_box)
            #img = filters.wear_ears(img.copy(), points, P, pose, pts_3d, roi_box)
            #img = filters.wear_nose(img.copy(), points, P, pose, pts_3d, roi_box)
        for P, pts_3d, roi_box in zip(Ps, pts_3ds, roi_boxes):
            img = filters.add_filters(img, P, pts_3d, roi_box, selected_filters, fancy_mode)
        return img


    #bgr
    def addFilters(self, frame, selected_filters, fancy_mode, bbox_steps='one', mirroring=True):
        frame = frame[:, :, ::-1]
        if mirroring:
            frame = frame[:, ::-1, :]
            bbox_steps = 'one'
        if len(selected_filters) == 0:
            return frame
        awsl = self.get_landmarks(frame)
        res = self.plot(frame, *awsl, selected_filters, fancy_mode)
        return res

'''
    def addFilters(self, frame, selected_filters):
        frame = frame[:, ::-1, :]
        awsl = self.get_landmarks(frame)
        res = self.plot(frame, *awsl, selected_filters)
        return res
'''


'''
selected_filters = {"eye":"glass", "ear":"rabbit_ear", "nose":"cat_nose"}
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img_name_list = ["cha1.jpg", "cha2.jpg", "ts1.jpg", "ts2.jpg", "ts3.jpg", "tc1.jpg"]
    img_name_list = ["tc2.png", "tc3.png", "tc4.png", "tc5.png", "b1.jpg", "b2.jpg"]
    img_name_list = ["cha1.jpg", "cha3.jpg", "cha4.jpg", "s.jpg", "t.jpg"]
    for name in img_name_list:
        img = cv2.imread("test_images/" + name)[:,:,::-1]
        print(img.shape)
        res = plot_landmarks(img, selected_filters)
        plt.imshow(res)
        plt.show()
        plt.imsave("test_results/"+name[:-4]+"_res.png", res)
'''