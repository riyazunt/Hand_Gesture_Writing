import cv2
import mediapipe as mp
import numpy as np
import math
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imagesize = 300

while True:
    success, img = cap.read()
    h, w, c = img.shape
    mphands = mp.solutions.hands
    hands = mphands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    framergb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        mp_drawing.draw_landmarks(img, handLMs, mphands.HAND_CONNECTIONS)

        imagewhite = np.ones((imagesize, imagesize, 3),np.uint8)*255
        imgcrop = img[abs(y_min):abs(y_max), abs(x_min):abs(x_max)]

        imagecropsize = imgcrop.shape
        # # imagewhite[0:imagecropsize[0], 0:imagecropsize[1]] = imgcrop

        aspratio = h/w
        if aspratio > 1:
            k = imagesize/h
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop, (wcal, imagesize))
            wgap = math.ceil((imagesize-wcal)/2)
            imagewhite[:, wgap:wcal+wgap] = imgresize   
        else:
            k = imagesize/w
            hcal = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop, (imagesize, hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imagesize-hcal)/2)
            imagewhite[hgap:hcal+hgap, :] = imgresize   

        cv2.imshow('cropimage', imgcrop)
        cv2.imshow('imagewhite', imagewhite)
        
    cv2.imshow('Image', img)
    k = cv2.waitKey(1)
    if k%256 == 27:
        cap.release()
        cv2.destroyAllWindows()

            