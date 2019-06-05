from PIL import Image
import cv2
import numpy as np


class Filter:
    def __init__(self, fname=None, fimg=None, fimg_path=None, ftype=None):
        self.name = fname
        self.image = fimg
        self.image_path = fimg_path
        self.type = ftype


filter_list = {}
filter_list["eye"] = ["glass", "eye1", "eye2", "eye3", "mojing"]
filter_list["ear"] = ["rabbit_ear", "ear1", "ear2", "ear3"]
filter_list["nose"] = ["cat_nose", "nose1", "nose2", "nose4", "nose5", "nose6"]


def padding2square(img):
    h, w = img.shape[:2]
    if h < w:
        delta = w - h
        # print(img.shape, np.full((delta, w, 3), 255).shape)
        img = np.concatenate((img, np.full((delta, w, 3), 255)), axis=0)
    elif h > w:
        delta = h - w
        img = np.concatenate((img, np.full((h, delta, 3), 255)), axis=1)
    return img


def getAllFilters():
    li = []
    # print(filter_list)
    for type, name_list in filter_list.items():
        for filter_name in name_list:
            img_path = "./filters_image/" + type + "/" + filter_name + "_show.png"
            img = cv2.imread("./filters_image/" + type + "/" + filter_name + "_show.png")
            img = padding2square(img)
            img = cv2.resize(img.astype(np.float32), (90, 90))
            li.append(Filter(filter_name, img.copy(), img_path, type))
    return li


all_filters = {}


class fff:
    def __init__(self, fimg, finfo):
        self.img = fimg
        self.info = finfo


for type, name_list in filter_list.items():
    for filter_name in name_list:
        str_card = type + filter_name
        if type == "eye":
            img = cv2.imread("./filters_image/" + type + "/" + filter_name + ".png", cv2.IMREAD_UNCHANGED)
            with open("./filters_image/eye/" + filter_name + ".txt", "r") as f:
                cx, cy, ratio = f.readline().split()
                cx, cy, ratio = int(cx), int(cy), float(ratio)
            all_filters[str_card] = fff(img, (cx, cy, ratio))
        elif type == "nose":
            img = cv2.imread("./filters_image/" + type + "/" + filter_name + ".png", cv2.IMREAD_UNCHANGED)
            with open("./filters_image/nose/" + filter_name + ".txt", "r") as f:
                cx, cy, ratio = f.readline().split()
                cx, cy, ratio = int(cx), int(cy), float(ratio)
            all_filters[str_card] = fff(img, (cx, cy, ratio))
        elif type == "ear":
            img1 = cv2.imread("./filters_image/ear/" + filter_name + "_left.png", cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread("./filters_image/ear/" + filter_name + "_right.png", cv2.IMREAD_UNCHANGED)
            with open("./filters_image/ear/" + filter_name + ".txt", "r") as f:
                cx1, cy1, ratio1 = f.readline().split()
                cx2, cy2, ratio2 = f.readline().split()
                cx1, cy1, cx2, cy2 = int(cx1), int(cy1), int(cx2), int(cy2)
                ratio1, ratio2 = float(ratio1), float(ratio2)
            all_filters[str_card] = fff((img1, img2), (cx1, cy1, ratio1, cx2, cy2, ratio2))


def add_filters(img, P, pts_3d, roi_box, selected_filters, fancy_mode=False):
    if "ear" in selected_filters and selected_filters["ear"] != None:
        img = wear_ears(img, P, pts_3d, roi_box, selected_filters["ear"], fancy_mode)
    if "eye" in selected_filters and selected_filters["eye"] != None:
        img = wear_glass(img, P, pts_3d, roi_box, selected_filters["eye"], fancy_mode)
    if "nose" in selected_filters and selected_filters["nose"] != None:
        img = wear_nose(img, P, pts_3d, roi_box, selected_filters["nose"], fancy_mode)
    return img


# points: 2d侧脸68点, P: matrix, pose: euler angle, pts_3d: 3d正脸68点 已经scale到图片尺寸,roi_box:
def wear_glass(img, P, pts_3d, roi_box, filter_name, fancy_mode):
    fimg = all_filters["eye" + filter_name].img
    cx, cy, ratio = all_filters["eye" + filter_name].info

    # fimg[:, :, [0, 2]] = fimg[:, :, [2, 0]] # bgr_alpha -> rgb_alpha
    fimg = cv2.cvtColor(fimg, cv2.COLOR_BGRA2RGBA)
    fh, fw = fimg.shape[:2]

    w = (pts_3d[0][16] - pts_3d[0][0]) * ratio
    h = w * (fh / fw)
    # z = (pts_3d[2][39] + pts_3d[2][27] + pts_3d[2][42])/3
    z = pts_3d[2][27]
    x0, y0 = pts_3d[0][27], pts_3d[1][27]

    ori_list = [[x0, y0, z], [x0 - w / 2, y0 + h * (cy / fh), z], [x0 - w / 2, y0 - h * (1 - cy / fh), z],
                [x0 + w / 2, y0, z]]
    new_list = [[cx, cy], [0, 0], [0, fh], [fw, cy]]
    # NOTE: lt, lb, rb,  rt
    # p_3d = np.array([[x0-w, y0+h, z], [x0-w, y0-h, z], [x0+w, y0-h, z], [x0+w, y0+h, z]])
    p_3d = np.array(ori_list)

    p_3d_homo = np.concatenate((p_3d, np.ones([p_3d.shape[0], 1])), axis=1)
    p_2d = (p_3d_homo @ P.T)[:, :2]

    p_2d[:, 1] = 121 - p_2d[:, 1]
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    p_2d[:, 0] = p_2d[:, 0] * scale_x + sx
    p_2d[:, 1] = p_2d[:, 1] * scale_y + sy

    p_fimg = np.array(new_list)
    M = cv2.getPerspectiveTransform(p_fimg[:4].astype(np.float32), p_2d[:4].astype(np.float32))
    glass = cv2.warpPerspective(fimg, M, img.shape[:2][::-1])

    transparent = glass[:, :, 3] != 0
    if not fancy_mode:
        img[:, :][transparent] = glass[:, :, :3][transparent]
        return img
    alpha = (glass[:, :, 3].astype(np.float32) / 255.0).reshape(glass.shape[0], glass.shape[1], 1)
    img[transparent] = (img[transparent] * (1.0 - alpha[transparent]) + glass[:, :, :3][transparent] * alpha[
        transparent]).astype(np.uint8)
    # img = (glass[:,:,:3] * alpha + img * (1-alpha)).astype(np.uint8)
    # glass = glass[:,:,:3] # convert to 3 channels
    # img[:, :][transparent] = glass[transparent]

    return img


def wear_ears(img, P, pts_3d, roi_box, filter_name, fancy_mode):
    fimg1, fimg2 = all_filters["ear" + filter_name].img
    # fimg1 = fimg1.copy()
    # fimg2 = fimg2.copy()
    cx1, cy1, ratio1, cx2, cy2, ratio2 = all_filters["ear" + filter_name].info

    fimg1 = cv2.cvtColor(fimg1, cv2.COLOR_BGRA2RGBA)  # bgr_alpha -> rgb_alpha
    fimg2 = cv2.cvtColor(fimg2, cv2.COLOR_BGRA2RGBA)  # bgr_alpha -> rgb_alpha
    # fimg1[:, :, [0, 2]] = fimg1[:, :, [2, 0]] # bgr_alpha -> rgb_alpha
    # fimg2[:, :, [0, 2]] = fimg2[:, :, [2, 0]] # bgr_alpha -> rgb_alpha

    for i in range(2):
        if i == 0:
            fh, fw = fimg1.shape[:2]
            cx, cy, ratio = cx1, cy1, ratio1
            face_h = pts_3d[1][8] - pts_3d[1][19]
            x0, y0 = pts_3d[0][19], pts_3d[1][19]
        else:
            fh, fw = fimg2.shape[:2]
            cx, cy, ratio = cx2, cy2, ratio2
            face_h = pts_3d[1][8] - pts_3d[1][24]
            x0, y0 = pts_3d[0][24], pts_3d[1][24]

        w = (pts_3d[0][16] - pts_3d[0][0]) * ratio
        z = (pts_3d[2][19] + pts_3d[2][24]) / 2
        y0 = y0 - face_h * 0.3
        h = w * (fh / fw)

        ori_list = [[x0, y0, z], [x0 + w * (1 - cx / fw), y0, z], [x0, y0 + h, z], [x0 + w * (1 - cx / fw), y0 + h, z]]
        new_list = [[cx, cy], [fw, cy], [cx, 0], [fw, 0]]
        p_3d = np.array(ori_list)

        p_3d_homo = np.concatenate((p_3d, np.ones([p_3d.shape[0], 1])), axis=1)
        p_2d = (p_3d_homo @ P.T)[:, :2]

        p_2d[:, 1] = 121 - p_2d[:, 1]
        sx, sy, ex, ey = roi_box
        scale_x = (ex - sx) / 120
        scale_y = (ey - sy) / 120
        p_2d[:, 0] = p_2d[:, 0] * scale_x + sx
        p_2d[:, 1] = p_2d[:, 1] * scale_y + sy

        p_fimg = np.array(new_list)
        M = cv2.getPerspectiveTransform(p_fimg[:4].astype(np.float32), p_2d[:4].astype(np.float32))
        if i == 0:
            ear = cv2.warpPerspective(fimg1, M, img.shape[:2][::-1])
        else:
            ear = cv2.warpPerspective(fimg2, M, img.shape[:2][::-1])

        transparent = ear[:, :, 3] != 0
        if fancy_mode:
            alpha = (ear[:, :, 3].astype(np.float32) / 255.0).reshape(ear.shape[0], ear.shape[1], 1)
            img[transparent] = (img[transparent] * (1.0 - alpha[transparent]) + ear[:, :, :3][transparent] * alpha[
                transparent]).astype(np.uint8)
        else:
            img[:, :][transparent] = ear[:, :, :3][transparent]

    return img


from time import time


def wear_nose(img, P, pts_3d, roi_box, filter_name, fancy_mode):
    fimg = all_filters["nose" + filter_name].img  # .copy()
    cx, cy, ratio = all_filters["nose" + filter_name].info

    fimg = cv2.cvtColor(fimg, cv2.COLOR_BGRA2RGBA)  # bgr_alpha -> rgb_alpha
    # fimg[:, :, [0, 2]] = fimg[:, :, [2, 0]] # bgr_alpha -> rgb_alpha
    fh, fw = fimg.shape[:2]

    w = (pts_3d[0][16] - pts_3d[0][0]) * ratio
    h = w * (fh / fw)
    z = pts_3d[2][30]
    x0, y0 = pts_3d[0][30], pts_3d[1][30]

    ori_list = [[x0, y0, z], [x0 - w / 2, y0 + h * (cy / fh), z], [x0 - w / 2, y0 - h * (1 - cy / fh), z],
                [x0 + w / 2, y0, z]]
    new_list = [[cx, cy], [0, 0], [0, fh], [fw, cy]]

    p_3d = np.array(ori_list)
    p_3d_homo = np.concatenate((p_3d, np.ones([p_3d.shape[0], 1])), axis=1)
    p_2d = (p_3d_homo @ P.T)[:, :2]

    p_2d[:, 1] = 121 - p_2d[:, 1]
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    p_2d[:, 0] = p_2d[:, 0] * scale_x + sx
    p_2d[:, 1] = p_2d[:, 1] * scale_y + sy

    p_fimg = np.array(new_list)
    M = cv2.getPerspectiveTransform(p_fimg[:4].astype(np.float32), p_2d[:4].astype(np.float32))
    nose = cv2.warpPerspective(fimg, M, img.shape[:2][::-1])

    transparent = nose[:, :, 3] != 0
    if not fancy_mode:
        img[:, :][transparent] = nose[:, :, :3][transparent]
        return img
    alpha = (nose[:, :, 3].astype(np.float32) / 255.0).reshape(nose.shape[0], nose.shape[1], 1)
    img[transparent] = (img[transparent] * (1.0 - alpha[transparent]) + nose[:, :, :3][transparent] * alpha[
        transparent]).astype(np.uint8)
    # transparent = glass[:,:,3] != 0
    # glass = glass[:,:,:3] # convert to 3 channels
    # img[:, :][transparent] = glass[transparent]

    return img

#def show_face(pts_3d):

    

'''
def calc_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

def calc_theta(x, y):
    return -math.degrees(math.asin(y / math.sqrt(x*x + y*y)))

def wear_glass_old(img, p, filter_name): #image, points_5, filter_image
    glass = cv2.imread("./filters_image/"+filter_name+".png", cv2.IMREAD_UNCHANGED)
    glass[:, :, [0, 2]] = glass[:, :, [2, 0]] # bgr -> rgb
    if p == []:
        return img
    h0, w0, d = glass.shape
    yjj = int(calc_dist(p[1], p[0]))
    w = int(yjj * 3)
    h = int((w/w0) * h0)
    nh = h + (w-h)//2 * 2 # w>h
    glass = cv2.resize(glass, (w, h), interpolation = cv2.INTER_CUBIC)
    padding = np.zeros(((w-h)//2, w, 4))
    glass = np.concatenate((padding, glass, padding), axis=0)
    #print("glass", glass.shape)

    theta = calc_theta(p[1][0]-p[0][0], p[1][1]-p[0][1])
    M = cv2.getRotationMatrix2D((w/2, nh/2), theta, 1)
    glass = cv2.warpAffine(glass, M, (w, nh))

    r1, r2 = max(0, p[1][1]-nh//2), min(img.shape[0], p[1][1]-nh//2+nh)
    c1, c2 = max(0, p[0][0]-yjj), min(img.shape[1], p[0][0]-yjj+w)
    glass = glass[:r2-r1, :c2-c1]
    transparent = glass[:,:,3] != 0
    glass = glass[:,:,:3] # convert to 3 channels
    img[r1:r2, c1:c2][transparent] = glass[transparent]
    return img
'''