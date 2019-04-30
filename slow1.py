import face_recognition as fr
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

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
    
while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1],::-1]

    face_locations = fr.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        faceImg = rgb_frame[top:bottom, left:right]
        faces_landmarks = fr.face_landmarks(faceImg)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        for landmarks in faces_landmarks:
            for landmark in landmarks.values():
                for (x, y) in landmark:
                    nx, ny = x+, y+left
                    cv2.circle(frame, (nx, ny), 2, (255, 255, 255), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()