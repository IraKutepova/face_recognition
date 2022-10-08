# модуль для создания папок с фото конкретного человека для обучения модели распознавания лиц
import cv2 as cv
import os

video = cv.VideoCapture(0)

facedetect = cv.CascadeClassifier('haar_face.xml')

count = 0
nameID = str(input('Enter Your Name:')).lower()

path = 'images/' + nameID

isExist = os.path.exists(path)

if isExist:
    print('Name Already Taken')
    nameID = str(input('Enter Your Name Again: '))
else:
    os.makedirs(path)

while True:
    ret,  frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        count += 1
        name = './images/' + nameID + '/' + str(count) + '.jpg'
        print('Creating images...' + name)
        cv.imwrite(name, frame[y:y+h, x:x+w])
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
    cv.imshow("WindowFrame", frame)
    cv.waitKey(1)
    if count > 500:
        break
video.release()
cv.destroyAllWindows()
