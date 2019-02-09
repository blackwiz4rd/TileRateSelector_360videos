from joblib import dump, load
import numpy as np
import os
import sklearn
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# NEURAL NETWORK MODEL FOR HDM prediction
ID = np.random.seed(1234)
quaternions = np.array([])
tiles = np.array([])
number_of_users = 1
## NOT USED TO TRAIN NN
filedir = 'results/uid-9f0512a7-3258-40bd-bd03-6d783bfb1e99/test0/Diving-2OzlksZBTiA/'
quaternions = np.append(quaternions, np.loadtxt(filedir+'Diving-2OzlksZBTiA_0.txt')[:-1,2:6])
tiles = np.append(tiles, np.load(filedir+'tiles.npy')[1:])

quaternions = quaternions.reshape(int(quaternions.size/4),4)
print('quaternions shape', quaternions.shape)
tiles = tiles.reshape(int(tiles.size/16),16)
print('tiles shape', tiles.shape)
print('number of users', number_of_users)

clf = load('nn.joblib')
## RESULTS
# Best NN training error: 0.020802
# Best NN test error: 0.034897

# ## TRAINING SET

# # load features, labels
X_validation, y_validation = quaternions, tiles
print(X_validation.shape, y_validation.shape)

# print(clf.predict(X_validation))
prob = clf.predict_proba(X_validation)
print(prob.shape)
print(prob)
np.save('prob',prob)

# plt.plot(np.arange(prob.shape[0]), np.arange(prob.shape[0]))
# plt.plot(prob[0,:],'.') # at t = 0

# PROBABILITY TRANSITIONS
plt.imshow(prob[200:300,:].T,interpolation='none',aspect='auto')
plt.xlabel('i-th tile')
plt.ylabel('j-th segment')
plt.colorbar()
plt.yticks(np.arange(1,16))
plt.show()
plt.imshow(prob[500:700,:].T,interpolation='none',cmap='binary',aspect='auto')
plt.show()
plt.imshow(prob[800:900,:].T,interpolation='none',cmap='binary',aspect='auto')
plt.colorbar()
plt.show()