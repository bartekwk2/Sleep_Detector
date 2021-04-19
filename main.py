import cv2
import face_recognition
import dlib
import csv
import matplotlib.pyplot as plt
from scipy import ndimage
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import os

from face_detectors import face_detection1, face_detection2, face_detection2,face_detection_when_partially_hidden
from csv_operation import read_csv_with_sampling, read_csv
from video_operation import cutAndScaleVideo,scaleImage
from speed_calculation import calc_speed,plot_motion,treshhold_motion,draw_treshhold_line,Average,speed_from_csv
from eye_detecion import detect_eye


directory = os.getcwd()


#Proba

detect_eye('/data/points/film1_points/LEye.csv', '/data/points/film1_points/REye.csv', '/data/films/film_1.mp4')



'''
#Detekcja twarzy
face_detection1("camera")
face_detection2("camera",directory+"/data/predictors/shape_predictor_68_face_landmarks.dat")
face_detection3("camera",directory+"/data/predictors/haarcascade_eye_tree_eyeglasses.xml")

#Przycięcie i wykadrowanie video
cutAndScaleVideo(directory+'/data/films/cut_film_1.mp4',0,10, 400, 800)

#Prędkość z plików CSV
speed_from_csv('LAnkle')

#Wyświetlanie punktów csv na video
capture = cv2.VideoCapture(directory+'/data/films/film_1.mp4')
versions = str(cv2.version).split('.')
fps = capture.get(cv2.CAP_PROP_FPS)
xREye,yREye,zREye,xOutREye,yOutREye = read_csv_with_sampling(directory+'/data/points/film1_points/REye.csv')
xLEye,yLEye,zLEye,xOutLEye,yOutLEye = read_csv_with_sampling(directory+'/data/points/film1_points/LEye.csv')


framesCount = len(xREye)
iterator = 0

while iterator < framesCount:
    _, frame = capture.read()
    cv2.circle(frame,(int(xREye[iterator]),int(yREye[iterator])),1,(0, 255, 0),2)
    cv2.circle(frame,(int(xLEye[iterator]),int(yLEye[iterator])),1,(0, 255, 0),2)
    cv2.imshow('Video', frame)
    iterator+=1
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break


#Kadrowanie twarzy punktami CSV części ciała i próba predictora do wykrycia punktów oczu

face_detection_when_partially_hidden()
'''