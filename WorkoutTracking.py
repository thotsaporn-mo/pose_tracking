# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 22:20:57 2021

@author: Namo
"""

import cv2
import numpy as np
import mediapipe as mp
import math

# import PoseModule as pm

cap = cv2.VideoCapture("trx_workout/trx_push_ups.mp4")
# cap = cv2.VideoCapture(0)


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose


count = 0
direction = 0
color = (255, 0, 255)


cap.set(3, 1280)
cap.set(4, 720)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('trainer.avi', fourcc, 20, (int(w),  int(h)))       # (1280,  720)



with mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.grab():                       # Check whether the next frame is none 
                                            # if capture from camera change to cap.open()
        
        success, img = cap.read()
        # img = cv2.resize(img, (1280, 720))
    
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # but pose use RGB
        img.flags.writeable = False                         # image is no longer writeable
        results = pose.process(img)                         # make prediction
        img.flags.writeable = True                          # image is now writeable
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)          # color conversion RGB 2 BGR
    
        
    
    
# ===================== Make black background =================================
        # height, width, ch = img.shape
        # black_img = np.zeros((height, width, 3), dtype=np.uint8)
# =========================== OR ==============================================     
        # black_img = np.zeros(img.shape, dtype=np.uint8)
# =============================================================================
    
    
        if results.pose_landmarks:
            # img = black_img             # comment this if you want original frame
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)


        
        
# ====================== Get landmarks to array ===============================
        lmList = []       
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
    
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
# =============================================================================
        
    
# ========================== Find angle of joint ==============================
        if len(lmList) != 0:
            x1, y1 = lmList[mpPose.PoseLandmark.LEFT_SHOULDER.value][1:]
            x2, y2 = lmList[mpPose.PoseLandmark.LEFT_ELBOW.value][1:]
            x3, y3 = lmList[mpPose.PoseLandmark.LEFT_WRIST.value][1:]
        
            # Calculate the Angle
            angle = math.degrees(math.atan2(y2-y1, x2-x1) -
                                  math.atan2(y2-y3, x2-x3))
            angle = np.abs(angle)
            if angle > 180:
                angle = 360 - angle
# =============================================================================

            
# ================== Convert Angle to Percent & Bar ===========================            
        if len(lmList) != 0:
            per = np.interp(angle, (90, 160), (0, 100))
            bar = np.interp(angle, (90, 160), (650, 100))
            # print(angle, per, bar)
    
            # Check for the curls
            if per == 100:
                color = (0, 255, 0)
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                color = (255, 0, 255)
                if direction == 1:
                    count += 0.5
                    direction = 0
            # print(count)
# =============================================================================   
    

# =========================== Draw Bar ========================================
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1050, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    color, 4)
# =============================================================================
    

# ========================== Draw Curl Count ==================================
        cv2.rectangle(img, (0,450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 10,
                    (255, 0, 0), 15)
# =============================================================================
    
        
        
# ==========================Write Video========================================
        out.write(img)
# =============================================================================
        
        
        
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
