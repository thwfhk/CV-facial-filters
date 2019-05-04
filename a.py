#%%
import face_recognition as fr 
import dlib
from imutils import face_utils
import cv2
import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time

glass = cv2.imread("glass.png")
plt.imshow(glass)
print(glass)
plt.show()


