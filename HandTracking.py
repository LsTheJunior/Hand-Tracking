import numpy as np
import cv2 
import mediapipe as mp
from cv2 import *

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(1)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():  
        ret, frame = cap.read()
        image =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results =  holistic.process(image)
        print(results.left_hand_landmarks)
        print(results.right_hand_landmarks)


        image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


        if not ret:
            break
        cv2.imshow("capture",image)
        cv2.waitKey(1)
        if cv2.getWindowProperty('capture', 4) < 1:
            break



cap.release()
cv2.destroyAllWindows()



    
    




        


