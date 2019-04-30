#%%
import face_recognition as fr 
import cv2
import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from time import time
import functools
def get_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        st = time()
        a = func(*args, **kwargs)
        print(func.__name__, "time:", time()-st)
        return a
    return wrapper

img = plt.imread("twh.jpg")
img2 = plt.imread("twh2.jpg")
@get_time
def plot_landmarks(img): #in rgb
    face_locations = fr.face_locations(img, model='hog')
    faces_landmarks = fr.face_landmarks(img, face_locations, model='large')

    print(face_locations)
    #return img
    for (top, right, bottom, left), face_landmarks in zip(face_locations, faces_landmarks):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmarks in faces_landmarks:
            for landmark in landmarks.values():
                for (x, y) in landmark:
                    cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
    return img

#plot_landmarks(img)
#plot_landmarks(img2)
mask = np.zeros((720, 1080, 3), dtype='bool')
mask[:, 180:900, :] = True
print(mask.shape, img.shape)
img = img[mask].reshape(720, 720, 3)
plt.imshow(img)
plt.show()

#plt.imshow(plot_landmarks(img))
#plt.show()



#%%
