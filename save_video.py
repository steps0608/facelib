import cv2
import numpy as np
import os
from os.path import isfile, join

# pathIn = '/home/minds/PycharmProjects/facelib/AgeGender/videosample6/'
pathIn = '/home/minds/Videos/videosample3/'
pathOut = '762957H01_1_18-TH-06-844__face2.mp4'
fps = 1.0


frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]  # for sorting the file names properly

files.sort(key=lambda x: x[5:-4])

files.sort()
frame_array = []

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]  # for sorting the file names properly
files.sort(key=lambda x: x[5:-4])

for i in range(len(files)):
    filename = pathIn + files[i]
    # reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    # inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)


for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])

out.release()