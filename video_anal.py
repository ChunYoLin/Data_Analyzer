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
cap = cv2.VideoCapture('../video_clipper/GRMN0726/10.mp4')
frame_id = 0
frame_lst = []
frame_show = []
while(cap.isOpened()):
    ret, frame = cap.read()
    print 'frame_id ',frame_id
    if frame != None:
        frame = frame[:, 886:1334]
        frame = cv2.resize(frame, (200, 200), interpolation = cv2.INTER_CUBIC)
        frame_show.append(frame)
        frame_lst.append(frame.reshape(-1))
    else:
        break
    frame_id += 1
    if frame_id == 500:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
frame_lst = np.asarray(frame_lst)
print frame_lst.shape
model = PCA(n_components = 2)
data = model.fit_transform(frame_lst) 
#  with open('data.pk', 'w') as f:
    #  pickle.dump(data, f)
data = data / 100.
for i in range(500):
    plt.subplot(121)
    plt.axis([-200, 200, -200, 200])
    plt.scatter(data[i, 0], data[i, 1])
    plt.title("project to 2D")
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(frame_show[i], cv2.COLOR_BGR2RGB))
    plt.title("original resized video")
    plt.savefig('/home/chunyo/SHARE/img/img%d.png'%(i))

img = cv2.imread('/home/chunyo/SHARE/img/img0.png')
height, width, layers = img.shape
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
fps = 30
video = cv2.VideoWriter()
video.open('/home/chunyo/SHARE/crop.mov', fourcc, fps, (width, height), True)
print video.isOpened()
for i in range(500):
    img = cv2.imread('/home/chunyo/SHARE/img/img%d.png'%(i))
    video.write(img)
cv2.destroyAllWindows()
video.release()
