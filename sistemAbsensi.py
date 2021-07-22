import cv2
import numpy as np
import face_recognition
import os

path = 'gambarAbsensi'
Gambar = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    Gambar.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def DataLatih(Gambar):
    encodeList = []
    for img in Gambar:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = DataLatih(Gambar)
print('Data sudah terlatih gan')

#################################################################
camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceFrame = face_recognition.face_locations(imgS)
    encodeFrame = face_recognition.face_encodings(imgS, faceFrame)

    for encodeFace,faceLoc in zip(encodeFrame,faceFrame):
        matches = face_recognition.compare_faces(DataLatih,encodeFace)
        faceDis = face_recognition.face_distance(DataLatih,encodeFace)
        print(faceDis)



# faceID = face_recognition.face_locations(imgAliga)[0]
# encodeAliga = face_recognition.face_encodings(imgAliga)[0]
# cv2.rectangle(imgAliga, (faceID[3],faceID[0]),(faceID[1],faceID[2]),(255,0,255),2)
# #print(faceID) ALIGA = (325, 304, 511, 118)

# faceIDCoba = face_recognition.face_locations(imgCoba)[0]
# encodeCoba = face_recognition.face_encodings(imgCoba)[0]
# cv2.rectangle(imgCoba, (faceIDCoba[3],faceIDCoba[0]),(faceIDCoba[1],faceIDCoba[2]),(255,0,255),2)

# # Uji Coba Hasil
# hasil = face_recognition.compare_faces([encodeAliga],encodeCoba)
# faceDistance = face_recognition.face_distance([encodeAliga],encodeCoba)
# print(hasil,faceDistance)
# cv2.putText(imgCoba,f'{hasil} {round(faceDistance[0],2)}',(20,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)