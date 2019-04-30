from PIL import Image
from detector import detect_faces
from visualization_utils import show_results
import matplotlib.pyplot as plt

img = Image.open('../../facial_filter/a.jpg') # modify the image path to yours
bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
plt.imshow(show_results(img, bounding_boxes, landmarks)) # visualize the results
plt.show()
print(bounding_boxes)
print(landmarks)
