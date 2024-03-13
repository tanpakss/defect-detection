import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

from decimal import Decimal, getcontext

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def change_res(width, height):
    cap.set(2, width)
    cap.set(2, height)

change_res(2000,2000)

while True:
    _, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Laplacian
    kernel = np.float32([[0,1,0],
            [1,-4,1],
            [0,1,0]])
    #Convolution
    res = cv2.filter2D(gray,0,kernel)
    #Guassian Blur
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    #Canny edge detection
    res1 = cv2.Canny(blurred,25,50)

    ksize= 3
    ker = np.ones((ksize,ksize),np.uint8)
    dilation = cv2.dilate(res1, kernel=ker,iterations=1)
    erode = cv2.erode(dilation, kernel=ker,iterations=1)
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of defect: {len(contours)}")

    i = 0
    j = 0
    contour_pos_x = []
    contour_pos_y = []
    box_width = 0
    box_length = 0
    aspect_ratio = []

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(frame, [box], 0, (0, 0, 255), 1)
        print("defect no. :", i+1)
        i+=1

        #print(box)  
        n = box.ravel() 

        
        contour_pos_x.append(n[0]) 
        contour_pos_y.append(n[1]) 

        box_width = np.linalg.norm(box[1] - box[2]) 
        box_length = np.linalg.norm(box[0] - box[1])
        
        aspect_ratio.append(box_width / box_length if box_length != 0 else 0)
    
    print("------------------")
    # time.sleep(2)

    for x in range(len(contours)):
        print("defect no. :", x+1 )
        #classify by using x and y
        if abs(aspect_ratio[x] - 1) < 0.40:
            print("Classification: DENT")
            cv2.putText(frame, 'DENT', (contour_pos_x[x], contour_pos_y[x]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            

        else:
            print("Classification: SCRATCH")
            cv2.putText(frame, 'SCRATCH', (contour_pos_x[x], contour_pos_y[x]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print("------------------")
        j+=1
    # time.sleep(0.5)
    
    #output of code in each part
    # cv2.imshow('output from gray',gray)
    # cv2.imshow('output from GaussianBlur',blurred)
    # cv2.imshow('output from Canny edge detection',res1)
    # cv2.imshow('output from dilate',dilation)
    # cv2.imshow('output from erodtion',erode)
    cv2.imshow('Original Image with Minimum Area Rectangle', frame)
    # key = cv2.waitKey(0)
    if cv2.waitKey(1) == 27: 
        break
cap.release()
cv2.destroyAllWindows()
