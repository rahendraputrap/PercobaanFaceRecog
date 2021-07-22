import cv2
import numpy as np
import face_recognition

imgAliga = face_recognition.load_image_file('gambar/aliga base.jpg')
imgAliga = cv2.cvtColor(imgAliga, cv2.COLOR_BGR2RGB)
imgCoba = face_recognition.load_image_file('gambar/milos.jpg')
imgCoba = cv2.cvtColor(imgCoba, cv2.COLOR_BGR2RGB)

faceID = face_recognition.face_locations(imgAliga)[0]
encodeAliga = face_recognition.face_encodings(imgAliga)[0]
cv2.rectangle(imgAliga, (faceID[3],faceID[0]),(faceID[1],faceID[2]),(255,0,255),2)
#print(faceID) ALIGA = (325, 304, 511, 118)

faceIDCoba = face_recognition.face_locations(imgCoba)[0]
encodeCoba = face_recognition.face_encodings(imgCoba)[0]
cv2.rectangle(imgCoba, (faceIDCoba[3],faceIDCoba[0]),(faceIDCoba[1],faceIDCoba[2]),(255,0,255),2)

# Uji Coba Hasil
hasil = face_recognition.compare_faces([encodeAliga],encodeCoba)
print(hasil)

cv2.imshow('Aliga', imgAliga)
cv2.imshow('Testing', imgCoba)
cv2.waitKey(0)