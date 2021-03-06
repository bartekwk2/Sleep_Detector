
import cv2
import face_recognition
import dlib
import csv
import matplotlib.pyplot as plt
from scipy import ndimage
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import os
import math
from csv_operation import read_csv_with_sampling,read_csv2,generate_labels,split_data
import matplotlib.pyplot as plt



def max_speed_for_second(dir):
    maxPerSecond = []
    maxPerSecondIndex = []
    allPartsSpeed,length = speed_body_parts(dir)

    for i in range(length-1):
        currentMax = 0
        currentIndexMax = 0

        for indexPart,part in enumerate(allPartsSpeed):
            if part[i]>=currentMax:
                currentMax = part[i]
                currentIndexMax = indexPart
        maxPerSecond.append(currentMax)
        maxPerSecondIndex.append(currentIndexMax)

    return maxPerSecond


def speed_body_parts(dir):
    bodyParts = ['LAnkle','RAnkle','LWrist','RWrist']
    allPartsSpeed = []

    for part in bodyParts:
        directory = os.getcwd()
        x,y,z,xOut,yOut= read_csv_with_sampling(directory+'/data/points/'+dir+'/'+part+'.csv')
        length = len(xOut)
        distance = []

        for i in range(length):
            if i < length-1:
                distance.append(math.hypot(xOut[i+1]-xOut[i],yOut[i+1]-yOut[i]))
        allPartsSpeed.append(distance)

    return allPartsSpeed,length

def speed_all_from_csv(name,treshhold,rows,columns,index):

        directory = os.getcwd()
        x,y,z,xOut,yOut= read_csv_with_sampling(directory+'/data/points/film1_points/'+name+'.csv')
        distance = []
        length = len(xOut)

        for i in range(length):
            if i < length-1:
                distance.append(math.hypot(xOut[i+1]-xOut[i],yOut[i+1]-yOut[i]))

        time,value = treshhold_motion(distance,treshhold)
        secV = list(range(0,length-1))

        plt.subplot(rows, columns, index)
        draw_treshhold_line(treshhold,False)
        plot_motion(list=distance,range=secV,name = name)

        return time,value

def speed_from_csv(name,treshhold):
    directory = os.getcwd()
    x,y,z,xOut,yOut= read_csv_with_sampling(directory+'/data/points/film1_points/'+name+'.csv')

    Vx = calc_speed(xOut)
    Vy = calc_speed(yOut)

    framesX = list(range(0, len(x)))
    framesY = list(range(0, len(y)))
    secX = list(range(0, len(xOut)))
    secY = list(range(0, len(yOut)))
    secVx = list(range(0, len(Vx)))
    secVy = list(range(0, len(Vy)))

    timeVx,valueVx = treshhold_motion(Vx, treshhold)
    timeVy,valueVy = treshhold_motion(Vy, treshhold)
    print("WARTO??CI X")
    print(valueVx)
    print(timeVx)
    print("WARTO??CI Y")
    print(valueVy)
    print(timeVy)

    plt.subplot(2, 1, 1)
    draw_treshhold_line(treshhold,True)
    plot_motion(list=Vx,range=secVx,name = name + " x")

    plt.subplot(2, 1, 2)
    draw_treshhold_line(treshhold,True)
    plot_motion(list=Vy,range=secVy,name= name + " y")
    plt.show()


def calc_speed(dist):
    vOut = []
    for i in range(len(dist)-1):
        vOut.append(dist[i+1]-dist[i])
    return vOut


def plot_motion(list,range,name):
    maxi = max(list)
    mini = min(list)
    plt.plot(range,list)
    plt.title(name)
    plt.xlabel('Frames')
    plt.ylabel('Values')
    plt.ylim([mini-200, maxi+200])


def treshhold_motion(listSpeed,treshhold):
    time =  []
    value = []
    for i in range(len(listSpeed)):
        if listSpeed[i]>treshhold or listSpeed[i]<-treshhold :
            time.append(i)
            value.append(listSpeed[i])
    return time,value

def draw_treshhold_line(treshhold,multiple):
    plt.axhline(y=treshhold, color='r', linestyle='-')
    if multiple:
        plt.axhline(y=-treshhold, color='r', linestyle='-')

def Average(lst):
    return sum(lst) / len(lst)


def plotValues(speed,eyeCount):
    plt.scatter(speed, eyeCount)
    plt.show()

def plotTwoSeriesValues(xSleepData,ySleepData,xAwakeData,yAwakeData):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(xAwakeData,yAwakeData, s=10, c='r', marker="o", label='second')
    ax1.scatter(xSleepData, ySleepData, s=10, c='b', marker="s", label='first')
    plt.show()



