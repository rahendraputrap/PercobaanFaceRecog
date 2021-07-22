import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

from pkg_resources import EntryPoint

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

def Absensi(nama):
    with open('absensi.csv','r+') as f:
        Daftar = f.readlines()
        DaftarNama = []
        for line in Daftar:
            entry = line.split(',')
            DaftarNama.append(entry[0])
        if nama not in DaftarNama:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{nama},{dtString}')

#Absensi('Putra')

encodeListKnow = DataLatih(Gambar)
print('Data sudah terlatih gan')

#################################################################
camera = cv2.VideoCapture(0)

while True:
    success, img = camera.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceFrame = face_recognition.face_locations(imgS)
    encodeFrame = face_recognition.face_encodings(imgS, faceFrame)

    for encodeFace,faceLoc in zip(encodeFrame,faceFrame):
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        print(faceDis)
        cocok = np.argmin(faceDis)

        # Raimu di tandai ng kene :v
        if matches[cocok]:
            nama = classNames[cocok].upper()
            print(nama)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,nama,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            Absensi(nama)
            

    cv2.imshow('Webcam',img)
    cv2.waitKey(0)



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