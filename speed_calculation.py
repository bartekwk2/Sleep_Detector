
import cv2
import face_recognition
import dlib
import csv
import matplotlib.pyplot as plt
from scipy import ndimage
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import os
from csv_operation import read_csv_with_sampling


def speed_from_csv(name):
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

    treshhold = 30
    timeVx,valueVx = treshhold_motion(Vx, treshhold)
    timeVy,valueVy = treshhold_motion(Vy, treshhold)
    print("WARTOŚCI X")
    print(valueVx)
    print(timeVx)
    print("WARTOŚCI Y")
    print(valueVy)
    print(timeVy)

    plt.subplot(2, 1, 1)
    draw_treshhold_line(treshhold)
    plot_motion(list=Vx,range=secVx,name = name + " x")

    plt.subplot(2, 1, 2)
    draw_treshhold_line(treshhold)
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

def draw_treshhold_line(treshhold):
    plt.axhline(y=treshhold, color='r', linestyle='-')
    plt.axhline(y=-treshhold, color='r', linestyle='-')

def Average(lst):
    return sum(lst) / len(lst)