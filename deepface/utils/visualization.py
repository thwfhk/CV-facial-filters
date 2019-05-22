import cv2

from .colors import get_random_color

def draw_bbox(npimg, bbox, color=(0, 255, 0)):
    cv2.rectangle(npimg, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), color, 2)

    if bbox.face_landmark is not None:
        for (x, y) in bbox.face_landmark:
            cv2.circle(npimg, (x, y), 1, color, -1)

    if bbox.score > 0.0:
        cv2.putText(npimg, "%s %.2f" % (('%s(%.2f):' % (bbox.face_name, bbox.face_score)) if bbox.face_name else '', bbox.score), (bbox.x, bbox.y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
        cv2.putText(npimg, "%s %.2f" % (('%s(%.2f):' % (bbox.face_name, bbox.face_score)) if bbox.face_name else '', bbox.score), (bbox.x - 1, bbox.y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)


def draw_bboxs(npimg, bboxs):
    for i, bbox in enumerate(bboxs):
        draw_bbox(npimg, bbox, color=get_random_color(i).tuple())
    return npimg
