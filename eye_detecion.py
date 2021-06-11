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
from sklearn.model_selection  import train_test_split
from sklearn import svm


def predict_svm(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = svm.SVC()
    clf.fit(X, y)




def detect_eye(pathLEye,pathREye,pathFilm,showData,):


    lAnkleCount = 0
    rAnkleCount = 0
    lWristCount = 0
    rWristCount = 0

    directory = os.getcwd()

    xLeftEye,yLeftEye,confLeftEye = read_csv(directory+pathLEye)
    xRightEye,yRightEye,confRightEye = read_csv(directory+pathREye)

    constAddingX = 40
    constAddingY = 20
    smallAdding = 15
    valuesPerSec = []

    capture = cv2.VideoCapture(directory+pathFilm)
    fps = capture.get(cv2.CAP_PROP_FPS)
    framesCount = len(xLeftEye)

    iterator = 0
    seconds = 0
    countsOpen = []

    while iterator < framesCount:
        _, frame = capture.read()

        #cv2.circle(frame,(int(xLeftEye[iterator]),int(yLeftEye[iterator])),1,(0, 255, 0),2)
        #cv2.circle(frame,(int(xRightEye[iterator]),int(yRightEye[iterator])),1,(0, 255, 0),2)

        crop_image_left_eye = frame[int(yLeftEye[iterator]-constAddingY) : int(yLeftEye[iterator]+constAddingY), 
        int(xLeftEye[iterator]-smallAdding ) :int(xLeftEye[iterator]+constAddingX)]

        crop_image_right_eye = frame[int(yRightEye[iterator]-constAddingY) : int(yRightEye[iterator]+constAddingY),
        int(xRightEye[iterator]-constAddingX) :int(xRightEye[iterator]+smallAdding)]

        cropped_eye = crop_image_left_eye

        if crop_image_left_eye.shape[0] > 0 and crop_image_left_eye.shape[1] > 0:
            cropped_eye = scaleImage(crop_image_left_eye,5)
        
        if crop_image_right_eye.shape[0] > 0 and crop_image_right_eye.shape[1] > 0:
            cropped_eye = scaleImage(crop_image_right_eye,5)


        #right_eye = scaleImage(crop_image_right_eye,5)

        #left_eye_edges = cv2.Canny(left_eye,35,50)

        gray = cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2GRAY)
        left_filtered = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)

        thresh = cv2.threshold(left_filtered, 25, 255, cv2.THRESH_BINARY_INV)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        diff = cv2.absdiff(dilate, thresh)
        howMany = np.where(diff>0)[0]
        #print(howMany.size)
        
        valuesPerSec.append(howMany.size)
        if iterator % 60 == 0 :
            countsOpenPerSec = Average(valuesPerSec)
            countsOpen.append(countsOpenPerSec)
            '''
            if showData :
                print(("--------- SEKUNDA {sec} ---------").format(sec = seconds))
                if countsOpenPerSec > 0 :
                     print("OCZY OTWARTE "+ str(countsOpenPerSec))
                for i in range(len(timesAbove)):
                    if seconds in timesAbove[i] :
                        print("RUCH "+decideBodyPart(i,lAnkleCount,rAnkleCount,lWristCount,rWristCount,valuesAbove))
            '''
            valuesPerSec.clear()
            seconds += 1
    
        # img = left_eye
        # img = cv2.medianBlur(img,5)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
        #                             param1=50,param2=30,minRadius=0,maxRadius=0)
        # print(circles)


        if showData:
            frameScaled = scaleImage(frame,0.5)
            cv2.namedWindow('Video1')
            cv2.moveWindow('Video1', 240,230)
            cv2.imshow('Video1', cropped_eye)

            cv2.namedWindow('Video2')
            cv2.moveWindow('Video2', 540,230)
            cv2.imshow('Video2', diff)

            cv2.namedWindow('Video3')
            cv2.moveWindow('Video3', 980,230)
            cv2.imshow('Video3', frameScaled)
            
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break


        iterator+=1

    return countsOpen





def Average(lst):
    return sum(lst) / len(lst)

def decideBodyPart(index,lAnkleCount,rAnkleCount,lWristCount,rWristCount,valuesAbove):
    title = ''

    if index == 0 :
        title = 'LAnkle '
        title+=str(valuesAbove[index][lAnkleCount])
        lAnkleCount+=1
    elif index == 1 :
        title = 'RAnkle '
        title+=str(valuesAbove[index][rAnkleCount])
        rAnkleCount+=1
    elif index == 2 :
        title = 'LWrist '
        title+=str(valuesAbove[index][lWristCount])
        lWristCount+=1
    else:
        title = "RWrist "
        title+=str(valuesAbove[index][rWristCount])
        rWristCount+=1

    return title
