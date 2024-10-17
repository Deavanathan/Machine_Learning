import cv2
import numpy as np
import face_recognition


img_elon = face_recognition.load_image_file('elon_musk_royal_society.jpg')
img_elon = cv2.cvtColor(img_elon,cv2.COLOR_BGR2RGB)

img_elon_test = face_recognition.load_image_file('elon_musk_test.jpg')
img_elon_test = cv2.cvtColor(img_elon_test,cv2.COLOR_BGR2RGB)

facLoc = face_recognition.face_locations(img_elon)[0]
encodeElon = face_recognition.face_encodings(img_elon)[0]
cv2.rectangle(img_elon,(facLoc[3],facLoc[0]),(facLoc[1],facLoc[2]),(255,0,255),2)

facLocTest = face_recognition.face_locations(img_elon_test)[0]
encodeElonTest = face_recognition.face_encodings(img_elon_test)[0]
cv2.rectangle(img_elon_test,(facLocTest[3],facLocTest[0]),(facLocTest[1],facLocTest[2]),(255,0,255),2)

#Apply linear SVM
results = face_recognition.compare_faces([encodeElon],encodeElonTest)
faceDis = face_recognition.face_distance([encodeElon],encodeElonTest)
print('Does the person is same?',results)
print('\n Face distance',faceDis)

cv2.putText(img_elon_test,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,225),2)

cv2.imshow('Elon musk',img_elon)
cv2.imshow('Elon musk Test',img_elon_test)

cv2.waitKey(0)
