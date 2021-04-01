import os
from os import listdir
from PIL import Image
import cv

path = os.path.join(os.getcwd(), '')
p14_video = ['2017-01-17_0323_VID0002aa.mp4', '2017-01-17_0323_VID0002ab.mp4']

frame_list_14 = []
for video in p14_video:
  cap = cv2.VideoCapture(video)
  print(cap)
  i = 0
  while(cap.isOpened()):
    ret, frame = cap.read() #ret stato lettura, se è false il video è finito. frame = matrice np che contiene il fotogramma
    if frame is not None:
      RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame_list_14.append(RGB)
    if frame is None:
      break
    i = i+1
  print(i)