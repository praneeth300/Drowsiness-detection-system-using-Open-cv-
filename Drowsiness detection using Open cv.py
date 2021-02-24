#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the required libraries
import cv2
import os
import numpy as np
import pandas as pd
from keras.models import load_model
import random

#pygame is a python package which is helpful for playing the sounds in the background when the driver tends to sleepy while driving'
from pygame import mixer

#Load the classifiers
#For the face detection
face=cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
#For the left eye detection
leye=cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
#For the right eye detection
reye=cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')


#label (Two class)
label=['Close','Open']


#Load the sound
mixer.init()
sound=mixer.Sound('alarm.wav')

#Load the model which helps to detect the faces
model=load_model('models/cnnCat2.h5')

#Open the default webcame
cap=cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
path = os.getcwd()
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while (True):
    #Read the camera
    ret,frame=cap.read()
    height,width=frame.shape[:2]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))

    left_eye = leye.detectMultiScale(gray)
    right_eye=reye.detectMultiScale(gray)

    cv2.rectangle(frame,(0,height-20),(200,height),(0,0,0),thickness=cv2.FILLED)
    

    #create a rectangle through out the face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100),1)

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye=cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye=cv2.resize(r_eye,(24,24))
        r_eye=r_eye/255
        r_eye=r_eye.reshape(24,24,-1)
        r_eye=np.expand_dims(r_eye,axis=0)
        rpred=model.predict_classes(r_eye)
        if(rpred[0]==1):
            label='Open'
        if(rpred[0]==0):
            label='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye=cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye=cv2.resize(l_eye,(24,24))
        l_eye=l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye=np.expand_dims(l_eye,axis=0)
        lpred=model.predict_classes(l_eye)
        if(lpred[0]==1):
            label='Open'
        if(lpred[0]==0):
            label='Closed'
        break
    #if both eye's are closed the score is increasing 
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20),font,1,(255,255,255),1,cv2.LINE_AA)
    #if both eye's are opened the score is decreasing
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20),font,1,(255,255,255),1,cv2.LINE_AA)

    if (score<0):
        score=0
    cv2.putText(frame,'Score:'+str(score),(100,height-20),font,1,(255,255,255),1,cv2.LINE_AA)

    if(score>13):
        #the person is tends to sleep while driving
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        
        #play a sound after the time limit is reached
        try:
            sound.play()

        except:
            pass
        if(thicc<16):
            thicc=thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2

        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
    cv2.imshow('frame',frame)
    #Exit when the 'q' key is pressed in the keyboard
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyWindow()


# In[ ]:




