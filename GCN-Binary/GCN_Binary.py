# import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.neighbors import KNeighborsClassifier
import math
import numpy as np
import pickle as pk

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
eps = 1e-20

# func to train a GCN
def trainGCN(data, labels):
  rows, cols = data.shape

  # initialize the W and Z matrix
  W = (np.random.random((597, 597)) - 0.5) / 10
  Z = (np.random.random((597, 1)) - 0.5) / 10

  # initialize total iterations & learning rate
  iterations = 2500
  lr = 0.1

  for i in range(iterations):

    X1 = np.dot(data, W)
    X11 = 1 / (1 + np.exp(-(X1)))
    X2 = np.dot(X11, Z)
    X22 = 1 / (1 + np.exp(-(X2)))

    # Backprop 
    L = (labels - X22) / rows
    L1 = L * (X22 * (1 - X22))
    L11 = L1.dot(Z.T)
    L2 = L11 * (X11 * (1 - X11))

    # Weight update
    Z += (lr * (X11.T.dot(L1)))
    W += (lr * (data.T.dot(L2)))

  return W, Z


# find the euclidian dist between two vectors
def findDist(vec1, vec2):
  dist = 0.0
  size = len(vec1)

  for i in range(size):
    dist += (vec1[i] - vec2[i])**2

  return math.sqrt(dist)


# find neighbourhood aggregation for given data points amongest them
def findNeighbourAggregation(data):
  # using 5 nearest neighbour + one self 
  # denominatior = sqrt(|N(v)|*|N(u)|) = 5
  # numerator = summation of neighbours

  totalRows, totalCols = data.shape
  finalAgg = []

  for i in range(totalRows):

    row1 = data[i]
    dist = [] 
    idx  = [] # store the 5 index which are nearest to the curr index

    for j in range(totalRows):

      if (i == j):
        continue

      d = findDist(row1, data[j])

      if (len(dist) < 5):
        dist.append(d)
        idx.append(j)
      elif(d < max(dist)):
        maxIdx = dist.index(max(dist))
        dist[maxIdx] = d
        idx[maxIdx]  = j

    # summing over the neighbour for curr index
    for k in range(5):
      row1 += data[idx[k]]

    # dividing by the denominator
    row1 = row1 / 5.0

    finalAgg.append(row1)

  return np.asarray(finalAgg)


# find neighbourhood aggregation for a particular node in the data
def neighAggForNode(data, node):
  # using 5 nearest neighbour + one self 
  # denominatior = sqrt(|N(v)|*|N(u)|) = 5
  # numerator = summation of neighbours

  totalRows, totalCols = data.shape
  finalAgg = []
  
  row1 = node
  dist = [] 
  idx  = [] # store the 5 index which are nearest to the curr index

  for i in range(totalRows):
    d = findDist(row1, data[i])

    # here data already contains the node so computing 6 (5 neigh + 1 self)
    if (len(dist) < 6):
      dist.append(d)
      idx.append(i)
    elif(d < max(dist)):
      maxIdx = dist.index(max(dist))
      dist[maxIdx] = d
      idx[maxIdx]  = i

  agg = np.zeros((1, totalCols))
  # summing over the neighbour for curr index
  for j in range(5):
    agg += data[idx[j]]

  # dividing by the denominator
  agg = agg / 5.0

  return np.asarray(agg)


def doAggregation(trainData, newData):
  totalRows, totalCols = newData.shape
  finalAgg = []

  for i in range(totalRows):
    agg = neighAggForNode(trainData, newData[i])
    finalAgg.append(agg)

  return np.asarray(finalAgg)


# if prob > 0.5 treat as class 1 else as class 0
def interpretResult(result):
  size = len(result)
  newResult = np.ones((size,1), int)
  for i in range(size):
    if (result[i] >= 0.5):
      newResult[i] = 1
    else:
      newResult[i] = 0

  return newResult


def computeAcc(W, Z, data, labels):
  X1 = np.dot(data, W)
  X11 = 1 / (1 + np.exp(-(X1)))
  X2 = np.dot(X11, Z)
  X22 = 1 / (1 + np.exp(-(X2)))

  return (metrics.accuracy_score(interpretResult(X22), labels),X22)

  
def interpretLabel(lbl):
  size = len(lbl)
  newLabel = c = np.ones((size,1), int)
  for i in range(size):
    if (lbl[i] == 'M'):
      newLabel[i] = 1
    else:
      newLabel[i] = 0

  return newLabel

def start_training(l1, l2, file):
    train_agg = []
    test_agg = []
    score_res = {}
    test_scores = []
    train_scores = []
    mx = 1001 
    for l in range(l1,l2+1):
        print ("Training for label " + str(l))
        trainFeatures = []
        trainLabels = []
        testLabels = []

        c = 0
        for img in img_list[l]:
            trainLabels.append('M')
            trainFeatures.append(train_features[img])
            c += 1
            if c > mx:
                break
        
        c = 0
        for img in neg_list[l]:
            trainLabels.append('B')
            trainFeatures.append(train_features[img])
            c += 1
            if c > mx:
                break

        for lbl in test_data[l]:
            if lbl == 1:
                testLabels.append('M')
            else:
                testLabels.append('B')
        
        trainFeatures = np.array(trainFeatures)
        for i in trainFeatures:
            i = i/(np.sqrt(np.sum(np.dot(i,i))))

        testFeatures = np.array(test_features)
        for i in testFeatures:
            i = i/(np.sqrt(np.sum(np.dot(i,i))))

        totalFeaturesVec = np.concatenate((trainFeatures, testFeatures), axis=0)

        # performing the neighbourhood aggregation
        print ('    Performing neighborhood aggregation')
        aggTrainFeatures = findNeighbourAggregation(trainFeatures)
        print ('    Performing test aggregation')
        # aggValFeatures   = doAggregation(totalFeaturesVec, valFeatures.values)
        aggTestFeatures  = doAggregation(totalFeaturesVec, testFeatures)
        print ('    Training using GCN')
        W, Z = trainGCN(aggTrainFeatures, interpretLabel(trainLabels))

        train_agg.append(aggTrainFeatures)
        test_agg.append(aggTestFeatures)
        trainAcc, trainRes = computeAcc(W, Z, aggTrainFeatures, interpretLabel(trainLabels))
        testAcc, testRes  = computeAcc(W, Z, aggTestFeatures, interpretLabel(testLabels))
        train_scores.append(trainRes)
        test_scores.append(testRes)

    f1 = open(file + '_train_agg.pkl', 'wb')
    f2 = open(file + '_test_agg.pkl', 'wb') 
    f3 = open(file + '_scores.pkl', 'wb')
    train = {}
    test = {}
    train_agg = np.array(train_agg)
    test_agg = np.array(test_agg)
    train['val'] = train_agg
    test['val'] = test_agg
    score_res['train'] = train_scores
    score_res['test'] = test_scores
    pk.dump(train, f1)
    pk.dump(test, f2)
    pk.dump(score_res, f3)
    f1.close()
    f2.close()
    f3.close()