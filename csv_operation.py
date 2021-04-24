import cv2
import face_recognition
import dlib
import csv
import matplotlib.pyplot as plt
from scipy import ndimage
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import os


def read_csv_with_sampling(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter =' ')
        x = []
        y = []
        z = []
        for row in reader:
            splited = row[0].split(';')
            x.append(float(splited[0]))
            y.append(float(splited[1]))
            z.append(float(splited[2]))
        xHelp = []
        xOut = []
        yHelp = []
        yOut = []
        for i in range(len(z)):
            if z[i] < 0.75:

                prevValX = 0
                nextValX = 0
                prevValY = 0
                nextValY = 0
                rangeVal = 2
                
                if i != 0 :
                    if (rangeVal+i) >= len(z):
                        rangeVal = len(z)-i-1
                    elif (i-rangeVal<0):
                        rangeVal = i
                    for j in range(1,rangeVal+1):
                        prevValX +=x[i-j]
                        nextValX +=x[i+j]
                        prevValY +=y[i-j]
                        nextValY +=y[i+j]
                    x[i] = (prevValX+nextValX)/(2*rangeVal)
                    y[i] = (prevValY+nextValY)/(2*rangeVal)
                else :
                    for j in range(1,rangeVal+1):
                        nextValX +=x[i+j]
                        nextValY +=y[i+j]
                    x[i] = (nextValX)/(rangeVal)
                    y[i] = (nextValY)/(rangeVal)

            if i % 60 == 0:
                if len(xHelp) != 0:
                    avgX = sum(xHelp) / len(xHelp)
                    xOut.append(avgX)
                    xHelp.clear()
                if len(yHelp) != 0:
                    avgY = sum(yHelp) / len(yHelp)
                    yOut.append(avgY)
                    yHelp.clear()
            xHelp.append(x[i])
            yHelp.append(y[i])

        return x,y,z,xOut,yOut

def read_csv(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter =' ')
        x = []
        y= []
        conf= []
        for row in reader:
            splited = row[0].split(';')
            x.append(float(splited[0]))
            y.append(float(splited[1]))
            conf.append(float(splited[2]))
        return x,y,conf

