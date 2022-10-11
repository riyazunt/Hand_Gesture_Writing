import cv2
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from tkinter import *
from random import randint
import threading
from pynput.keyboard import Key, Controller
 
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

keyboard = Controller()

offset = 20
imgSize = 300
 
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'DELETE', 'SPACE', 'CLEAR']
text = ''

t = None

def press():
    global t
    keyboard.press('s')
    keyboard.release('s') 
    t = threading.Timer(2, press)
    t.start()
    
press()

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    out = img.copy()
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

#         cv2.rectangle(imgOutput, (x - offset, y - offset-50),(x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
#         cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        cv2.rectangle(imgOutput, (x-offset, y-offset),(x + w+offset, y + h+offset), (0, 0, 255), cv2.FILLED)
        cv2.putText(out, text, (50,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                
    key = cv2.waitKey(5)
     
    if key == ord('s') and hands:
        
        alpha = 0.01
        mask = imgOutput.astype(bool)
        out[mask] = cv2.addWeighted(img, alpha, imgOutput, 1 - alpha, 0)[mask]
        cv2.imshow("ImageO", out)
        
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, indsex = classifier.getPrediction(imgWhite, draw=False)
        
        
        def update(text):
            if labels[index] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                text = text+labels[index]
            elif labels[index] == 'DELETE' and text:
                text = text[:-1]
            elif labels[index] == 'SPACE':
                text = text+' '
            elif labels[index] == 'CLEAR':
                text = ''
            else:
                pass
                
            return text

        text = update(text)
        
    else:    
        cv2.imshow("ImageO", out)
        
    k = cv2.waitKey(5)
    if k == ord('q'):
        t.cancel()
        break
        
cap.release()
cv2.destroyAllWindows()   