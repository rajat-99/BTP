import numpy as np
import scipy.io 
import pickle
import os
import csv

path_to_train_features = 'espgame_data_vggf_pca_train.txt'
path_to_test_features = 'espgame_data_vggf_pca_test.txt'

train_features = []
test_features = []

f = open(path_to_train_features, 'r', encoding="utf-8")
for l in f:
    s = l.split(',')
    features = np.array(s,dtype='float32')
    train_features.append(features)

f = open(path_to_test_features, 'r', encoding="utf-8")
for l in f:
    s = l.split(',')
    features = np.array(s,dtype='float32')
    test_features.append(features)

train_features = np.array(train_features)
test_features = np.array(test_features)

np.savetxt('train_vggf.csv', train_features, delimiter=',')
np.savetxt('test_vggf.csv', test_features, delimiter=',')