from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import cv2
import os
import pickle
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
img_lst = []
with open('./yolo_train.txt') as f:
    for img_id, path in enumerate(f.read().splitlines()):
        print 'handle image', img_id
        if img_id == 3000:
            break
        img = cv2.imread(path)
        img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_CUBIC)
        img_lst.append(img.reshape(-1))
cap = cv2.VideoCapture('../video_clipper/GRMN0726/10.mp4')
frame_id = 0
frame_lst = []
frame_show = []
while(cap.isOpened()):
    ret, frame = cap.read()
    print 'handle frame',frame_id
    if frame != None:
        frame = frame[:, 886:1334]
        frame = cv2.resize(frame, (200, 200), interpolation = cv2.INTER_CUBIC)
        frame_show.append(frame)
        frame_lst.append(frame.reshape(-1))
    else:
        break
    frame_id += 1
    if frame_id == 1000:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

data_lst = img_lst + frame_lst
data_lst = np.asarray(data_lst)
model = PCA(n_components = 2)
model.fit(data_lst) 
data_img = model.transform(img_lst)
data_frame = model.transform(frame_lst)
for pt in data_img:
    plt.scatter(pt[0], pt[1], c = 'r')
for pt in data_frame:
    plt.scatter(pt[0], pt[1], c = 'b')
plt.savefig('compare.png')
