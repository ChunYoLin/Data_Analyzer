from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import cv2
import os
import sys
import pickle
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import progressbar

SET = 'kitti'
ts_vec = []
with open(sys.argv[1]) as f:
    for idx, row in enumerate(f.read().splitlines()):
        row = row.split()
        ts_vec.append(row[0:])                
ts1_vec = []
with open(sys.argv[2]) as f:
    for idx, row in enumerate(f.read().splitlines()):
        row = row.split()
        ts1_vec.append(row[0:])                
tr_vec = []
with open('./yolo_' + SET + '_traindata.txt') as f:
    for idx, row in enumerate(f.read().splitlines()):
        row = row.split()
        tr_vec.append(row[0:])                
        
vec = tr_vec + ts_vec + ts1_vec
tr_vec = np.asarray(tr_vec, dtype = np.float32)
ts_vec = np.asarray(ts_vec, dtype = np.float32)
ts1_vec = np.asarray(ts1_vec, dtype = np.float32)
vec = np.asarray(vec, dtype = np.float32)
model = PCA(n_components = 2)
model.fit(vec) 
tr_data = model.transform(tr_vec)
ts_data = model.transform(ts_vec)
ts1_data = model.transform(ts1_vec)
for pt in tr_data:
    plt.scatter(pt[0], pt[1], c = 'r')
for pt in ts_data:
    plt.scatter(pt[0], pt[1], c = 'b')
for pt in ts1_data:
    plt.scatter(pt[0], pt[1], c = 'g')
plt.savefig(SET + '_compare.png')
