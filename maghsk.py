# %%
# from deepface import get_detector as mbn
import deepface as mbn
import dlib

from MEOW3DDFA import a_3ddfa
import filters


class twh:
    def __init__(self, mo='gpu'):
        self.detector = mbn.get_detector()
        self.landmarkor = a_3ddfa.my3ddfa(mo)


    def get_landmarks(self, img, bbox_steps):  # in rgb
        floc = self.detector.detect(img)
        face_locations = []
        faces_landmarks = []
        faceRects = []
        for (x1, y1, x2, y2, p) in floc:
            face_locations.append(list(map(int, (y1, x2, y2, x1))))
            faceRects.append(dlib.rectangle(int(x1), int(y1), int(x2), int(y2)))

        # NOTE: using 3ddfa
        faces_landmarks, Ps, poses, pts_3ds, roi_boxes = self.landmarkor.meow_landmarks(img, faceRects, bbox_steps)

        return face_locations, faces_landmarks, Ps, poses, pts_3ds, roi_boxes


    def plot(self, img, face_locations, faces_landmarks, Ps, poses, pts_3ds, roi_boxes, selected_filters, fancy_mode):
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
            img = filters.add_filters(img, P, pts_3d, roi_box, selected_filters, fancy_mode)
        return img

    # 使用rgb，进行了镜面处理
    def addFilters(self, frame, selected_filters, fancy_mode, bbox_steps='one', mirroring=True):
        if mirroring:
            frame = frame[:, ::-1, :]
            bbox_steps = 'one'
        if len(selected_filters) == 0:
            return frame[:,:,::-1]
        awsl = self.get_landmarks(frame, bbox_steps)
        frame = frame[:,:,::-1]
        res = self.plot(frame, *awsl, selected_filters, fancy_mode)
        return res
