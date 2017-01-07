from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import os
import sys
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIR = sys.argv[1]
all_data = []
data_dict = {}
for fname in os.listdir(DIR):
    with open(DIR + '/' + fname) as f:
        data = []
        for vec in f.read().splitlines():
            vec = vec.split()
            data.append(vec)
    all_data += data
    data_dict[os.path.splitext(fname)[0]] = np.asarray(data, np.float32)
all_data = np.asarray(all_data, np.float32)
model = PCA(n_components = 2)
model.fit(all_data) 
plt.title(sys.argv[2])
colormap = plt.cm.gist_ncar 
colorst = [colormap(i) for i in np.linspace(0, 0.9, len(data_dict))]

for idx, (k, v) in enumerate(data_dict.iteritems()):
    draw_data = model.transform(v)
    for ptidx, pt in enumerate(draw_data):
        plt.scatter(pt[0], pt[1], c = colorst[idx])
classes = data_dict.keys()
recs = []
for i in range(len(data_dict)):
        recs.append(mpatches.Rectangle((0,0), 1, 1, fc = colorst[i]))
lgd = plt.legend(recs, classes, loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.savefig(sys.argv[2] + '.png', bbox_extra_artists = (lgd,), bbox_inches = 'tight')
