#!/usr/bin/env python
# coding: utf-8

# # Real time video demo for Face Emotion Recognition

# In[1]:


#import cv2
#import deepface
#from deepface import DeepFace
#import matplotlib.pyplot as plt

#faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#cap = cv2.VideoCapture(1)
#Check if the webcam is opened correctly
#if not cap.isOpened():
#    cap = cv2.VideoCapture(0)
#if not cap.isOpened():
#    raise IOError("Cannot open Webcam")
    
#while True:
#    ret, frame = cap.read() #read one image from a video
#    
#    result = {}
#    #result = DeepFace.analyze(frame, actions = emotion)
#    result = DeepFace.analyze(frame, actions =['emotion'])
    
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
#    #print(faceCascade.empty())
#    faces = faceCascade.detectMultiScale(gray,1.1,4)

#   #Draw rectangle arounf the faces
#    for(x, y, w, h) in faces:
#       cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#    font = cv2. FONT_HERSHEY_SIMPLEX
#    #Use putText() method for inserting text on video
#    cv2.putText(frame,
#               result[0]['dominant_emotion'],
#               (50, 50),
#               font, 1,
#               (0, 0, 255),
#                2,
#                cv2.LINE_4)
#    cv2.imshow('Demo video',frame)
    
#    if cv2.waitKey(2) & 0xFF == ord('q'):
#        break
        
#cap.release()
#cv2.destroyAllWindows() 


# In[ ]:


import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained face and emotion classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model("emotion_model.hdf5")

# Define the emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Set the video capture device (change to 0 if using built-in camera)
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI) from the grayscale image
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        # Normalize the ROI grayscale image and convert it to a 4D tensor
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict the emotion label for the ROI grayscale image using the pre-trained model
        preds = emotion_model.predict(roi_gray)[0]
        label = EMOTIONS[preds.argmax()]

        # Draw the predicted emotion label and a rectangle around the face in the original color image
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the frame with the emotion detection results
    cv2.imshow("Emotion Detection", frame)

    # Wait for a key press and exit if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
#### colab code


# In[ ]:




