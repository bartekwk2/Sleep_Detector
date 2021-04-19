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


def face_detection_when_partially_hidden():
    
    directory = os.getcwd()
    versions = str(cv2.__version__).split('.')
    xLeftEar,yLeftEar,confLeftEar = read_csv(directory+'/data/points/film1_points/LEar.csv')
    xRightEar,yRightEar,confRightEar = read_csv(directory+'/data/points/film1_points/REar.csv')
    xNose,yNose,confNose = read_csv(directory+'/data/points/film1_points/Nose.csv')
    xLShoulder,yShoulder,confLShoulder = read_csv(directory+'/data/points/film1_points/RShoulder.csv')
    xLEye,yLEye,confLEye= read_csv(directory+'/data/points/film1_points/LEye.csv')

    framesCount = len(xLeftEar)
    iterator = 0

    differenceEars = []
    maxBetweenEars=0

    differenceNoseLEar = []
    maxBetweenNoseLEar=0
    minBetweenNoseLEar = 0

    differenceNoseREar = []
    maxBetweenNoseREar=0
    minBetweenNoseREar = 0

    yEars = 0
    yShoulderMy = 0
    yEyes = 0

    while iterator < framesCount:

        if xLeftEar[iterator] != 0 and xRightEar[iterator] !=0 :
            differenceNowEars = abs(xLeftEar[iterator]-xRightEar[iterator])
            differenceEars.append(differenceNowEars)
            maxBetweenEars = max(differenceEars)
        
        if xLeftEar[iterator] != 0  and xNose[iterator] !=0  :
            differenceNowNoseLEar= abs(xLeftEar[iterator]-xNose[iterator])
            differenceNoseLEar.append(differenceNowNoseLEar)
            maxBetweenNoseLEar = max(differenceNoseLEar)
            minBetweenNoseLEar = min(differenceNoseLEar)

        if xRightEar[iterator] != 0 and xNose[iterator] !=0 :
            differenceNowNoseREar= abs(xRightEar[iterator]-xNose[iterator])
            differenceNoseREar.append(differenceNowNoseREar)
            maxBetweenNoseREar = max(differenceNoseREar)
            minBetweenNoseREar = min(differenceNoseREar)

        if yEars == 0 :
            yEars = yLeftEar[iterator]
            if yEars == 0 :
                yEars = yRightEar[iterator]

        if yShoulderMy == 0 :
            yShoulderMy = yShoulder[iterator]

        if yEyes == 0 :
            yEyes = yLEye[iterator]

                
                
        if(iterator == 900):
            break
        iterator+=1

    iterator = 0
    capture = cv2.VideoCapture(directory+'/data/films/film_1.mp4')
    fps = capture.get(cv2.CAP_PROP_FPS)
    predictor = dlib.shape_predictor(directory+"/data/predictors/shape_predictor_81_face_landmarks.dat")
    constAdding = 0

    while iterator < framesCount:

        _, frame = capture.read()
        additionalOne = 0
        additionalTwo = 0
        UpperLeftPointX = 0
        UpperLeftPointY = 0
        LowerRightPointX = 0
        LowerRightPointY = 0

        if xLeftEar[iterator] == 0 :
            differenceLEarNoseNow = abs(xRightEar[iterator]-xNose[iterator])
            percentage = differenceLEarNoseNow/maxBetweenEars
            additionalOne = percentage*60
            output = (1 - percentage)* (maxBetweenEars/2)
            UpperLeftPointX = output+xNose[iterator]+additionalTwo
            UpperLeftPointY = yShoulderMy+10
        else :
            additionalOne = 0
            UpperLeftPointX = xLeftEar[iterator]+additionalTwo
            UpperLeftPointY = yShoulderMy+10

        upperPosition = (yEyes-yShoulderMy) + yEyes -60

        if xRightEar[iterator] ==0 :
            differenceREarNoseNow = abs(xLeftEar[iterator]-xNose[iterator])
            percentage = differenceREarNoseNow/maxBetweenEars
            additionalTwo = percentage *60
            output = (1 - percentage)* (maxBetweenEars/2)
            LowerRightPointX = xNose[iterator]-output-additionalOne
            LowerRightPointY = upperPosition
        else :
            additionalTwo = 0
            LowerRightPointX = xRightEar[iterator]-additionalOne
            LowerRightPointY = upperPosition

        #cv2.circle(frame,(int(UpperLeftPointX),int(UpperLeftPointY)),1,(0, 255, 0),2)
        #cv2.circle(frame,(int(LowerRightPointX),int(LowerRightPointY)),1,(0, 255, 0),2)
        #cv2.rectangle(frame, (int(UpperLeftPointX+constAdding), int(UpperLeftPointY+constAdding)), (int(LowerRightPointX-constAdding), int(LowerRightPointY-constAdding)), (0, 255, 0), 2)
        
        crop_image = frame[int(LowerRightPointY-constAdding) : int(UpperLeftPointY+constAdding), int(LowerRightPointX-constAdding) :int(UpperLeftPointX+constAdding )]
        drect = dlib.rectangle(int(LowerRightPointX+constAdding), int(LowerRightPointY+constAdding),int(UpperLeftPointX-constAdding), int(UpperLeftPointY-constAdding))

        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        landmarks = predictor(grey,drect)
        for  i in range(1,68): 
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow('Video', frame)
        iterator+=1
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break


def face_detection1(source):
    if source == 'camera':
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(source)
    escape = 27
    fps = capture.get(cv2.CAP_PROP_FPS)
    face_locations = []
    while capture.isOpened():
        ret, frame = capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(rgb_frame)
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0,
                                                                255), 2)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == escape:
            break
    return capture.release() and cv2.destroyWindow()



def face_detection2(filename,predictor):
    if filename == 'camera':
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(source)

    escape = 27
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor)

    while cam.isOpened():
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            landmarks = predictor(gray, face)

            for i in range(0, 68):
                if 35 < i < 48:
                    cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 3, (255, 0, 0), -1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == escape:
            break


def face_detection3(filename,cascadeSource,scaleFactor = 1.1):

    if filename == 'camera':
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(source)

    escape = 27

    cascade  = cv2.CascadeClassifier(cascadeSource)

    while cam.isOpened():
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

        for (x, y, w, h) in faces_rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == escape:
            break