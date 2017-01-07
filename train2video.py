import sys
import numpy as np
import cv2
import os
import pickle
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import progressbar
with open(sys.argv[1]) as f:
    path_lst = f.read().splitlines()
    img = cv2.imread(path_lst[0])
    height, width, layers = img.shape
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    fps = 30
    video = cv2.VideoWriter()
    video.open(sys.argv[2], fourcc, fps, (448, 448), True)
    print video.isOpened()
    bar = progressbar.ProgressBar(maxval = progressbar.UnknownLength)
    for path in bar(path_lst):
        img = cv2.imread(path)
        img = cv2.resize(img, (448, 448))
        video.write(img)
cv2.destroyAllWindows()
video.release()
