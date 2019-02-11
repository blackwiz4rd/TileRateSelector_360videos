from joblib import dump, load
import numpy as np
import os
import sklearn
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from feature_selector import *

# NEURAL NETWORK MODEL FOR HDM prediction
ID = np.random.seed(1234)
## USED TO RUN NN
filedir = '../results/uid-9f0512a7-3258-40bd-bd03-6d783bfb1e99/test0/Diving-2OzlksZBTiA/'
quaternions = np.loadtxt(filedir+'Diving-2OzlksZBTiA_0.txt')[:,2:6]
timestamps = np.loadtxt(filedir+'Diving-2OzlksZBTiA_0.txt')[:,0]
tiles = np.load(filedir+'tiles.npy')
number_of_users = 1

quaternions = quaternions.reshape(int(quaternions.size/4),4)
tiles = tiles.reshape(int(tiles.size/16),16)

print('quaternions shape', quaternions.shape)
print('tiles shape', tiles.shape)
print('timestamps shape', timestamps.shape)
print('number of users', number_of_users)

clf = load('nn.joblib')

# ## TRAINING SET

# assign features and labels
quaternions, tiles = feature_selector_training(quaternions, tiles, timestamps)
tiles = tiles.reshape(int(tiles.size/16),16)
quaternions = quaternions.reshape(tiles.shape[0],5*4)

X_validation, y_validation = quaternions, tiles
print("x_val", X_validation.shape)
print("y_val", y_validation.shape)

# print(clf.predict(X_validation))
prob = clf.predict_proba(X_validation)
print(prob.shape)
np.save('prob',prob)
np.save('y_validation',y_validation)

validation_error = 1. - clf.score(X_validation,y_validation) #ADD YOUR CODE

print ("NN validation error: %f" % validation_error)

# PROBABILITY TRANSITIONS
plt.subplot(1, 2, 1)
plt.imshow(prob.T,interpolation='none',aspect='auto')
plt.ylabel('i-th tile')
plt.xlabel('j-th segment')

plt.subplot(1, 2, 2)
plt.imshow(y_validation.T,interpolation='none',aspect='auto') # cmap='binary'
plt.ylabel('i-th tile')
plt.xlabel('j-th segment')
plt.colorbar()
plt.show()