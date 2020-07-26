import numpy as np
import os
import pickle as pk
import csv
import scipy.io 

mat = scipy.io.loadmat('traindata.mat')
mat = mat['m']
mat = np.array(mat)
data = np.transpose(mat)
num_classes = 268
num_img = 18689
adj = np.zeros(shape=(num_classes, num_classes))
neg_list = []
img_list = []
label_list = []
diff = []

for i in range(num_classes):
    for j in range(num_classes):
        if(i != j):
            adj[i][j] = np.dot(data[i], data[j])

for i in range(num_classes):
    imgs = []
    for j, lbl in enumerate(data[i]):
        if(lbl == 1):
            imgs.append(j)
    img_list.append(imgs)

for i in range(num_img):
    lbls = []
    for j in range(num_classes):
        if(mat[i][j] == 1):
            lbls.append(j)
    label_list.append(lbls)

img_cnt = np.array(adj)
for i in range(num_classes):
    s = np.sum(img_cnt[i])
    l = len(img_list[i])
    c = 0
    z = 0
    for j in range(num_classes):
        if img_cnt[i][j] != 0:
            if c == 0:
                img_cnt[i][j] = np.round((l*img_cnt[i][j])/s)
            else:
                if z%3 == 0:
                    img_cnt[i][j] = np.round((l*img_cnt[i][j])/s)
                else:
                    img_cnt[i][j] = np.ceil((l*img_cnt[i][j])/s)
                z += 1
            c ^= 1

ratio = []

for i in range(num_classes):
    negs = []
    for j in range(num_classes):
        cnt = 0
        for k in img_list[j]:
            if cnt >= img_cnt[i][j]:
                break
            if mat[k][i] == 0:
                if k not in negs:
                    negs.append(k)
                    cnt += 1
    neg_list.append(negs)
    diff.append(len(img_list[i]) - len(negs))
    ratio.append(diff[i]/len(img_list[i]))

res = {}
res['adj'] = adj
res['img_list'] = img_list
res['neg_list'] = neg_list
res['label_list'] = label_list
# file = open('esp_data2.pkl', 'wb')
# pk.dump(res, file)
# file.close()