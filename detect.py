import numpy as np
import cv2


def car_detect(img,size=0.5):
  
    classifier=cv2.CascadeClassifier('data/cascade.xml')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    check=classifier.detectMultiScale(gray,1.3,5)

    if check is ():
        print("NO CAR FOUND")

    for (x,y,w,h) in check:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.imshow('DETECTED',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()





#TEST DATA

st='TestImages/test-'
for i in range(170):
    ne=st+str(i)+'.pgm'
    print(ne)
    image=cv2.imread(ne)
    car_detect(image)

st='TestImages_Scale/test-'
for i in range(108):
    ne=st+str(i)+'.pgm'
    print(ne)
    image=cv2.imread(ne)
    car_detect(image)
