import cv2
import face_recognition
import dlib
import csv
import matplotlib.pyplot as plt
from scipy import ndimage
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import os


def cutAndScaleVideo(file,start,end, width, height):
    fileName = file.replace('.mp4','')
    ffmpeg_extract_subclip(file, start, end, targetname=fileName +"_cutted.mp4")
    escape = 27

    #fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter(fileName + '_cutted_scaled.mp4', fourcc, 20, (width, height))
    capture = cv2.VideoCapture(fileName + '_cutted.mp4')
    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            sky = frame[300:1300, 200:900]
            resized_frame = cv2.resize(sky, (width, height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            out.write(resized_frame)
            cv2.imshow('', resized_frame)
        else:
            break
        key = cv2.waitKey(1)
        if key == escape:
            break
    capture.release()
    out.release()

def scaleImage(image,scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized