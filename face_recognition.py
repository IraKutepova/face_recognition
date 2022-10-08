import cv2 as cv
import numpy as np
# существующий каскад классов для определения лиц
haar_cascade = cv.CascadeClassifier('haar_face.xml')
# люди, для определения которых, обучена модель в модуле face_train.py
people = ['irina', 'misha2']
# подключаем обученную модель для использования
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
# включаем веб-камеру 
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
# выбор шрифта
font = cv.FONT_HERSHEY_COMPLEX
# бесконечный цикл для обработки потока фреймов    
while True:
    # фрейм как исходная картинка
    success, imgOriginal = cap.read()
    # перевод в оттенки серого
    gray = cv.cvtColor(imgOriginal, cv.COLOR_BGR2GRAY)
    # определение лиц на фрейме
    faces_rect = haar_cascade.detectMultiScale(gray, 1.3, 5)
    # полученные координаты обрабатываем
    for (x,y,w,h) in faces_rect:
        # подготовка для распознования
        crop_img = gray[y:y+h, x:x+w]
        test_img = cv.resize(crop_img, (224, 224))
        
        # получаем индекс человека на фрейме и %уверенности в определении
        label, confidence = face_recognizer.predict(test_img)
        # если нужно вывести текстом в консоли
        # print(f'Label = {people[label].title()} with a confidence of {confidence}') 

        # добавка текста в углу экрана
        # cv.putText(imgOriginal, str(people[label].title()), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        # добавка текста и рамки
        cv.rectangle(imgOriginal, (x,y), (x+w, y+h), (50,0,100), thickness=2)
        cv.rectangle(imgOriginal, (x,y), (x+w,y+h), (50,0,100), 3)
        cv.rectangle(imgOriginal, (x,y-40), (x+w,y), (50,0,100),-2)
        cv.putText(imgOriginal, str(people[label].title()),(x,y-10), font, 0.75, (255,255,255),1, cv.LINE_AA)
        cv.putText(imgOriginal,str(round(confidence, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv.LINE_AA)
    # показ видео с добавлением рамки и текста при определении лица
    cv.imshow("Result",imgOriginal)
    # ожидания горячей клавиши для прекращения программ
    k=cv.waitKey(1)
    if k==ord('q'):
        break

# очистка
cap.release()
cv.destroyAllWindows()
