import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img = cv2.imread("twh.jpg") 
def brick(img):
    img = cv2.bilateralFilter(img, 15, 75, 75)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.astype("float32")
    img[:,:,1] *= 1.4
    img = np.minimum(img, 255)
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    img = (img / 256.0).astype("float32")
    step = 15
    h, w, d = img.shape
    nh, nw = w//step, h//step
    img2 = np.zeros_like(img)
    val = (1.3, 1.1, 1.2) #b,g,r
    for r1 in range(0, h, step):
        for c1 in range(0, w, step):
            r2 = min(r1+step, h)
            c2 = min(c1+step, w)
            for dim in range(d):
                img2[r1:r2, c1:c2, dim] = img[r1:r2, c1:c2, dim].mean() * val[dim]
    return img2

cv2.imshow("a", brick(img))
cv2.waitKey(0)
#exit()

video_capture = cv2.VideoCapture(0)
cv2.namedWindow('meow')

mask = np.zeros((720, 1280, 3), dtype='bool')
mask[:, 280:1000, :] = True
while(True):
    ret, frame = video_capture.read()
    frame = frame[mask].reshape(720, 720, 3)
    cv2.imshow('meow', brick(frame))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break