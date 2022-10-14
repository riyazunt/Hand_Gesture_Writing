import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import threading
import time
import sys 
import tensorflow as tf

# st.set_page_config(layout="wide")

st.title('Hand Gesture Writing')
run = st.checkbox('Click check box to start camera')

frame_window = st.image([])

detector = HandDetector(maxHands=1)

model = tf.keras.models.load_model('model/transferlearned_gesture.h5')
# model = Classifier("model/keras_model.h5", "model/labels.txt")

offset = 20
imgSize = 224
 
# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'DELETE', 'SPACE', 'CLEAR']    
randLabels = ['A','B','K','L','M','N','O','P','Q','R','S','T','C','U','V','W','X','Y','Z','DELETE','SPACE','CLEAR','D','E','F','G','H','I','J']
text = ''
error = ''

s = st.text(text)
er = st.text(error)

def predict(*args):
    
    global bool_val, timer
    bool_val = True
    time.sleep(0.05)
    bool_val = False
    timer = threading.Timer(3, predict)
    timer.start()    

predict() 

cam = cv2.VideoCapture(0) 

while run:
  
    try:
        ret, img = cam.read()
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
            # cv2.putText(out, text, (50,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1) 

        if bool_val and hands:

            alpha = 0.01
            mask = imgOutput.astype(bool)
            out[mask] = cv2.addWeighted(img, alpha, imgOutput, 1 - alpha, 0)[mask]
            frame_window.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

                norm=imgWhite/255.0
                reshaped=np.reshape(norm,(1,224,224,3))
                reshaped = np.vstack([reshaped])

#                 pred, index = model.getPrediction(imgWhite, draw=False)

                pred = model.predict(reshaped)
                index = np.argmax(pred)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

                norm=imgWhite/255.0
                reshaped=np.reshape(norm,(1,224,224,3))
                reshaped = np.vstack([reshaped])

#                 pred, index = model.getPrediction(imgWhite, draw=False)

                pred = model.predict(reshaped)
                index = np.argmax(pred)


            def update(text):
                if randLabels[index] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    text = text+randLabels[index]
                elif randLabels[index] == 'DELETE' and text:
                    text = text[:-1]
                elif randLabels[index] == 'SPACE':
                    text = text+' '
                elif randLabels[index] == 'CLEAR':
                    text = ''
                else:
                    pass

                return text

            text = update(text)
            # s.write(f"<h1 style='text-align: center; color:red;'>{text}</h1>", unsafe_allow_html=True)
            s.write(f"<h1 style='color:black;'>{text}</h1>", unsafe_allow_html=True)
            error = 'reading success'
            er.write(f"<p style='color:green;'>{error}</p>", unsafe_allow_html=True) 


        else:  
            frame_window.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    except:
        error = 'Either camera turned off or an error occured in reading image, \
               \nrestart camera and keep the hand within window'
        er.write(f"<p style='color:red;'>{error}</p>", unsafe_allow_html=True)
        break

        
cam.release() 
timer.cancel()
frame_window.image([])
st.write('stopped')     
