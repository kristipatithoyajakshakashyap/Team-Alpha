import cv2
from deepface import DeepFace
import numpy as np
import simpleaudio as sa
import time 


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #processing it for our project

cap = cv2.VideoCapture(0)
Happyname = "Happy.wav"
Sadname = "Sad.wav"
Angryname = "angry.wav"
Neutralname = "Neutral.wav"
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        img =  cv2.rectangle(frame, (x,y), (x+w, y+h) ,(0,0,255),1)
        try:
          analyze = DeepFace.analyze(frame,actions=['emotion'])  #same thing is happing here as the previous example, we are using the analyze class from deepface and using ‘frame’ as input
          print(analyze['dominant_emotion'])  #here we will only go print out the dominant emotion also explained in the previous example
          if (analyze['dominant_emotion'] == 'happy'):
            wave_obj = sa.WaveObject.from_wave_file(Happyname)
            play_obj = wave_obj.play()
            time.sleep(2)
          elif (analyze['dominant_emotion'] == 'sad'):
            wave_obj = sa.WaveObject.from_wave_file(Sadname)
            play_obj = wave_obj.play()
            time.sleep(2)
          elif (analyze['dominant_emotion'] == 'angry'):
            wave_obj = sa.WaveObject.from_wave_file(Angryname)
            play_obj = wave_obj.play()
            time.sleep(2)
          else:
            wave_obj = sa.WaveObject.from_wave_file(Neutralname)
            play_obj = wave_obj.play()
            time.sleep(2)
        except:
          print("no face")
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
      break
cap.release()