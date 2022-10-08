# Обучение модели

import os
import cv2 as cv
import numpy as np

people = ['irina', 'misha2']
DIR = r'C:\Ira\Face recognition\images'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

# function for train
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        # print(path)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+w, x:x+h]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done --------------')
# print(f'The length of features = {len(features)}')
# print(f'The length of labels = {len(labels)}')
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on features and labels
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)