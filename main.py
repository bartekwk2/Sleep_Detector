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
from csv_operation import read_csv_with_sampling, read_csv,generate_labels,write_csv
from video_operation import cutAndScaleVideo,scaleImage
from speed_calculation import calc_speed,plot_motion,treshhold_motion,draw_treshhold_line,Average,speed_from_csv,speed_all_from_csv
from eye_detecion import detect_eye








counts_open = detect_eye('/data/points/film3_points/LEye.csv', '/data/points/film3_points/REye.csv', '/data/films/film_3.mp4',False)
print(counts_open)
write_csv(counts_open,'dziecko')



#speed_from_csv("LAnkle",30)



'''
directory = os.getcwd()
timesAboveLAnkle,valuesAboveLAnkle = speed_all_from_csv('LAnkle',30,2,2,1)
timesAboveRAnkle,valuesAboveRAnkle = speed_all_from_csv('RAnkle',30,2,2,2)
timesAboveLWrist,valuesAboveLWrist = speed_all_from_csv('LWrist',30,2,2,3)
timesAboveRWrist,valuesAboveRWrist = speed_all_from_csv('RWrist',30,2,2,4)
plt.show()

timesAll = [timesAboveLAnkle,timesAboveRAnkle,timesAboveLWrist,timesAboveRWrist]
valuesAll = [valuesAboveLAnkle,valuesAboveRAnkle,valuesAboveLWrist,valuesAboveRWrist]

eyeCounts = detect_eye('/data/points/film1_points/LEye.csv', '/data/points/film1_points/REye.csv', '/data/films/film_1.mp4',timesAll,valuesAll,True)

print(eyeCounts)
'''
'''
#Detekcja twarzy
face_detection1("camera")
face_detection2("camera",directory+"/data/predictors/shape_predictor_68_face_landmarks.dat")
face_detection3("camera",directory+"/data/predictors/haarcascade_eye_tree_eyeglasses.xml")

#Przycięcie i wykadrowanie video
cutAndScaleVideo(directory+'/data/films/cut_film_1.mp4',0,10, 400, 800)

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