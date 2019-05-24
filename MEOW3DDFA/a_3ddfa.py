#!/usr/bin/env python3
# coding: utf-8
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import dlib
import scipy.io as sio
import argparse
import torch.backends.cudnn as cudnn

from .utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
from . import mobilenet_v1
from .utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from .utils.cv_plot import plot_pose_box
from .utils.estimate_pose import parse_pose
from .utils.render import get_depths_image, cget_depths_image, cpncc
from .utils.paf import gen_img_paf

STD_SIZE = 120

device = 'gpu'

# 1. load pre-tained model
checkpoint_fp = './MEOW3DDFA/models/phase1_wpdc_vdc.pth.tar'
arch = 'mobilenet_1'

checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

model_dict = model.state_dict()
# because the model is trained by multiple gpus, prefix module should be removed
for k in checkpoint.keys():
    model_dict[k.replace('module.', '')] = checkpoint[k]
model.load_state_dict(model_dict)
if device == 'gpu':
    cudnn.benchmark = True
    model = model.cuda()
model.eval()

# 2. load dlib model for face detection and landmark used for face cropping
dlib_landmark_model = "./dlib_data/shape_predictor_68_face_landmarks.dat"
face_regressor = dlib.shape_predictor(dlib_landmark_model)

def meow_landmarks(img_ori, rects, use_landmarks=True, bbox_steps='one'):
    img_ori = img_ori[:,:,::-1] #rgb->bgr

    # 3. forward
    #tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]) #mean ans std!!!
    
    pts_res = []
    Ps = []  # Camera matrix collection
    poses = []  # pose collection, [todo: validate it]
    pts_3ds = [] # 三维正脸的68点
    roi_boxes = []
    for rect in rects:
        # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
        if use_landmarks:
            # - use landmark for cropping 
            pts = face_regressor(img_ori, rect).parts()
            pts = np.array([[pt.x, pt.y] for pt in pts]).T
            roi_box = parse_roi_box_from_landmark(pts) #TODO: to read it 
        else:
            # - use detected face bbox
            # use rects as bbox ang generate roi_bbox
            bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
            roi_box = parse_roi_box_from_bbox(bbox) # square *1.58

        img = crop_img(img_ori, roi_box)

        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0) # use transform!!!
        with torch.no_grad():
            if device == 'gpu':
                input = input.cuda()
            param = model(input)  # NOTE: 输入图像是resize后的ROI
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # 68 pts
        pts68, pts_3d = predict_68pts(param, roi_box)
        # NOTE: pts68:原始图像坐标系上, pts_3d:三维正脸上

        # two-step for more accurate bbox to crop face
        if bbox_steps == 'two':
            roi_box = parse_roi_box_from_landmark(pts68) #2333 roi_box again
            img_step2 = crop_img(img_ori, roi_box)
            img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img_step2).unsqueeze(0)
            with torch.no_grad():
                if device == 'gpu':
                    input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cuda().numpy().flatten().astype(np.float32)
                else:
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            pts68 = predict_68pts(param, roi_box) #2333 predict again

        pts_res.append(pts68)
        P, pose, scale_f = parse_pose(param) # NOTE: get cm and angle
        Ps.append(P)
        poses.append(pose)

        pts_3d *= scale_f
        pts_3ds.append(pts_3d)

        roi_boxes.append(roi_box)

        ''' show 3d
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = pts_3d[0]
        Y = pts_3d[1]
        Z = pts_3d[2]
        ax.scatter(X, Y, Z)
        plt.show()
        '''

    '''
    img_pose = plot_pose_box(img_ori, Ps, pts_res)
    #img_pose = qwq(img_ori, Ps, pts_res)
    import matplotlib.pyplot as plt
    plt.imshow(img_pose[:,:,::-1])
    plt.show()
    '''
    return pts_res, Ps, poses, pts_3ds, roi_boxes

if __name__ == '__main__':
    import matplotlib.pyplot as plt
