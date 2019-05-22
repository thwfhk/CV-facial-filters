# -*- coding: utf-8 -*-
import math

import cv2
import numpy as np

try:
    from itertools import zip_longest
except:
    from itertools import izip_longest as zip_longest

from deepface.confs.conf import DeepFaceConfs


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def rotate_dot(point, mat):
    px, py = point

    qx = mat[0][0] * px + mat[0][1] * py + mat[0][2]
    qy = mat[1][0] * px + mat[1][1] * py + mat[1][2]
    return int(qx), int(qy)


def roundint(v):
    return int(round(v))


def tag_faces(faces, result, threshold):
    for face_idx, face in enumerate(faces):
        face.face_feature = result['feature'][face_idx]
        name, score = result['name'][face_idx][0]
        if score < threshold:
            continue
        face.face_name = name
        face.face_score = score

    return faces


def faces_to_rois(npimg, faces, roi_mode='recognizer_vgg'):
    rois = []
    for face in faces:
        roi = get_roi(npimg, face, roi_mode=roi_mode)
        rois.append(roi)
    return rois


def get_roi(img, face, roi_mode):
    """
    :return: target_size 크기의 Cropped & Aligned Face Image
    """
    rpy, node_point = landmark_to_pose(face.face_landmark, img.shape)
    roll = rpy[0]
    if abs(roll) > math.pi / 2.0:
        roll = 0.0  # TODO ?

    height, width = img.shape[:2]

    new_w, new_h = (abs(math.sin(roll) * height) + abs(math.cos(roll) * width),
                    abs(math.sin(roll) * width) + abs(math.cos(roll) * height))
    new_w = roundint(new_w)
    new_h = roundint(new_h)
    mat = cv2.getRotationMatrix2D((height / 2, width / 2), -1 * roll * 180.0 / math.pi, 1.0)
    (tx, ty) = (roundint((new_w - width) / 2), roundint((new_h - height) / 2))
    mat[0, 2] += tx
    mat[1, 2] += ty
    dst = cv2.warpAffine(img, mat, dsize=(new_w + tx * 2, new_h + ty * 2))

    aligned_points = []
    if face.face_landmark is not None:
        for x, y in face.face_landmark:
            new_x, new_y = rotate_dot((x, y), mat)
            aligned_points.append((new_x, new_y))

        min_x = min(aligned_points, key=lambda x: x[0])[0]
        max_x = max(aligned_points, key=lambda x: x[0])[0]
        min_y = min(aligned_points, key=lambda x: x[1])[1]
        max_y = max(aligned_points, key=lambda x: x[1])[1]

        aligned_w = max_x - min_x
        aligned_h = max_y - min_y
        crop_y_ratio = float(DeepFaceConfs.get()['roi'][roi_mode]['crop_y_ratio'])
        center_point = ((min_x + max_x) / 2, min_y * crop_y_ratio + max_y * (1.0 - crop_y_ratio))
        image_size = int(
            max(aligned_w, aligned_h) * DeepFaceConfs.get()['roi'][roi_mode]['size_ratio'])  # TODO : Parameter tuning?
    else:
        min_x, min_y = rotate_dot((face.x, face.y), mat)
        max_x, max_y = rotate_dot((face.x + face.w, face.y + face.h), mat)

        aligned_w = max_x - min_x
        aligned_h = max_y - min_y
        center_point = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        image_size = int(max(aligned_w, aligned_h) * DeepFaceConfs.get()['roi'][roi_mode]['size_ratio'])

    crop_x1 = roundint(center_point[0] - image_size / 2)
    crop_y1 = roundint(center_point[1] - image_size / 2)
    crop_x2 = roundint(center_point[0] + image_size / 2)
    crop_y2 = roundint(center_point[1] + image_size / 2)

    cropped = dst[max(0, crop_y1):min(new_h, crop_y2), max(0, crop_x1):min(new_w, crop_x2)]
    pasted = np.zeros((image_size, image_size, 3), np.uint8)

    start_x = 0 if crop_x1 > 0 else -crop_x1
    start_y = 0 if crop_y1 > 0 else -crop_y1
    crop_w = min(cropped.shape[1], pasted.shape[1] - start_x)
    crop_h = min(cropped.shape[0], pasted.shape[0] - start_y)
    try:
        pasted[start_y:start_y + crop_h, start_x:start_x + crop_w] = cropped[:crop_h, :crop_w]  # TODO
    except:
        print(crop_y_ratio, (1.0 - crop_y_ratio), roll, pasted.shape, cropped.shape, 'min', min_x, max_x, min_y, max_y,
              'imgsize', image_size, start_x, start_y, crop_w, crop_h)
        print(center_point)
        # crop_y_ratio set 0.3667256819925064
        # 0.0 (128, 128, 3) (249, 128, 3) min 76 166 101 193 imgsize 128 0 129 128 -1
        # (121.0, -65.31315823528763)

    return pasted


def landmark_to_pose(landmark, image_shape):
    image_points = np.array([
        landmark[33],  # (359, 391),  # Nose tip
        landmark[8],  # (399, 561),  # Chin
        landmark[36],  # (337, 297),  # Left eye left corner
        landmark[45],  # (513, 301),  # Right eye right corne
        landmark[48],  # (345, 465),  # Left Mouth corner
        landmark[54],  # (453, 469)  # Right mouth corner
    ], dtype='double')

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    center = (image_shape[1] / 2, image_shape[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rotation, jacobian = cv2.Rodrigues(rotation_vector)
    translation = np.array(translation_vector).reshape(-1, 1).T[0]

    permutation_marker_to_ros = np.array((
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=np.float64)
    permutation_camera_to_ros = np.array((
        (0.0, 0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=np.float64)

    relation_cv = np.concatenate((
        np.concatenate((rotation, translation.reshape(3, 1)), axis=1),
        np.array((0.0, 0.0, 0.0, 1.0)).reshape(1, 4)
    ), axis=0)
    relation = permutation_camera_to_ros.dot(relation_cv)
    relation = relation.dot(np.linalg.inv(permutation_marker_to_ros))

    if success:
        nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                       translation_vector, camera_matrix, dist_coeffs)
        return rotationMatrixToEulerAngles(relation), nose_end_point2D
    else:
        return None, None


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def feat_distance_cosine(feat1, feat2):
    similarity = np.dot(feat1 / np.linalg.norm(feat1, 2), feat2 / np.linalg.norm(feat2, 2))
    return similarity


def feat_distance_l2(feat1, feat2):
    feat1_norm = feat1 / np.linalg.norm(feat1, 2)
    feat2_norm = feat2 / np.linalg.norm(feat2, 2)
    similarity = 1.0 - np.linalg.norm(feat1_norm - feat2_norm, 2) / 2.0
    return similarity
