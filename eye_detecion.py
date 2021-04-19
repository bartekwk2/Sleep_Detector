import cv2
import face_recognition
import dlib
import csv
import matplotlib.pyplot as plt
from scipy import ndimage
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import os
from csv_operation import  read_csv
from video_operation import scaleImage



def detect_eye(pathLEye,pathREye,pathFilm):

    directory = os.getcwd()

    xLeftEye,yLeftEye,confLeftEye = read_csv(directory+pathLEye)
    xRightEye,yRightEye,confRightEye = read_csv(directory+pathREye)

    constAddingX = 40
    constAddingY = 20
    smallAdding = 15

    capture = cv2.VideoCapture(directory+pathFilm)
    fps = capture.get(cv2.CAP_PROP_FPS)
    framesCount = len(xLeftEye)
    iterator = 0

    while iterator < framesCount:
        _, frame = capture.read()

        #cv2.circle(frame,(int(xLeftEye[iterator]),int(yLeftEye[iterator])),1,(0, 255, 0),2)
        #cv2.circle(frame,(int(xRightEye[iterator]),int(yRightEye[iterator])),1,(0, 255, 0),2)

        crop_image_left_eye = frame[int(yLeftEye[iterator]-constAddingY) : int(yLeftEye[iterator]+constAddingY), 
        int(xLeftEye[iterator]-smallAdding ) :int(xLeftEye[iterator]+constAddingX)]

        crop_image_right_eye = frame[int(yRightEye[iterator]-constAddingY) : int(yRightEye[iterator]+constAddingY),
        int(xRightEye[iterator]-constAddingX) :int(xRightEye[iterator]+smallAdding)]

        left_eye = scaleImage(crop_image_left_eye,5)
        right_eye = scaleImage(crop_image_right_eye,5)

        #left_eye_edges = cv2.Canny(left_eye,35,50)

        gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        left_filtered = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)

        thresh = cv2.threshold(left_filtered, 25, 25, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        diff = cv2.absdiff(dilate, thresh)

        howMany = np.where(diff>0)[0]
        print(howMany.size)

        # img = left_eye
        # img = cv2.medianBlur(img,5)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
        #                             param1=50,param2=30,minRadius=0,maxRadius=0)
        # print(circles)

        iterator+=1
        cv2.namedWindow('Video1')
        cv2.moveWindow('Video1', 240,230)
        cv2.imshow('Video1', left_eye)

        cv2.namedWindow('Video2')
        cv2.moveWindow('Video2', 540,230)
        cv2.imshow('Video2', diff)
        
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break