import os 
import csv
import numpy as np
import pickle as pk
import scipy.io

path_train_features = 'train_vggf.csv'
path_test_features = 'test_vggf.csv'

train_features = np.genfromtxt(path_train_features, delimiter=',') 
test_features = np.genfromtxt(path_test_features, delimiter=',')
res = pk.load(open('esp_data2.pkl', 'rb'))
adj = res['adj']
img_list = res['img_list']
neg_list = res['neg_list']
label_list = res['label_list']
res_test = pk.load(open('esp_test.pkl', 'rb'))
test_data = res_test['test']

trainFeatures = []
trainLabels = []
testFeatures = []
testLabels = []

for img in img_list[0]:
    trainLabels.append('M')
    trainFeatures.append(train_features[img])

for img in neg_list[0]:
    trainLabels.append('B')
    trainFeatures.append(train_features[img])

for lbl in test_data[0]:
    if lbl == 1:
        testLabels.append('M')
    else:
        testLabels.append('B')

trainFeatures = np.array(trainFeatures)

res = {}
res['trainFeatures'] = trainFeatures
res['testFeatures'] = test_features
res['trainLabels'] = trainLabels
res['testLabels'] = testLabels
file = open('test1.pkl', 'wb')
pk.dump(res, file)
file.close()