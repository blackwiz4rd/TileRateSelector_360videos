#load the required packages
from joblib import dump, load
import numpy as np
import os
import sklearn
from sklearn.linear_model import LinearRegression
from feature_selector import *

# NEURAL NETWORK MODEL FOR HDM prediction
ID = np.random.seed(1234)
quaternions = np.array([])
tiles = np.array([])
number_of_users = 0
for userid in os.listdir('../results'):
	if userid[0]!=".":
		filedir = '../results/'+userid+'/test0/Diving-2OzlksZBTiA/'
		if os.path.isfile(filedir+'Diving-2OzlksZBTiA_0.txt'):
			data = np.loadtxt(filedir+'Diving-2OzlksZBTiA_0.txt')
			data1 = np.load(filedir+'tiles.npy')

			temp_quaternions, temp_tiles = feature_selector_training(data[:,2:6], data1, data[:,0])
			quaternions = np.append(quaternions, temp_quaternions)
			tiles = np.append(tiles, temp_tiles)
			number_of_users = number_of_users + 1

tiles = tiles.reshape(int(tiles.size/16),16)
print('tiles shape', tiles.shape)
quaternions = quaternions.reshape(tiles.shape[0],5*4) # trick for training
print('quaternions shape', quaternions.shape)
			
print('number of users', number_of_users)
m_training = int(quaternions.shape[0]*0.8)
# exit()
# ## TRAINING SET

# # load features, labels
X_train, y_train = quaternions[:m_training,:], tiles[:m_training,:]
# print(X_train.shape, y_train.shape)

# load features, labels
X_test, y_test = quaternions[m_training:,:], tiles[m_training:,:]
print(X_test.shape, y_test.shape)

# #TRAIN NEURAL NETWORK
regr = LinearRegression()
regr.fit(X_train, y_train)

training_error = 1. - regr.score(X_train,y_train) #ADD YOUR CODE
test_error = 1. - regr.score(X_test,y_test) #ADD YOUR CODE

print ("Best reg training error: %f" % training_error)
print ("Best reg test error: %f" % test_error)

# save the trained model
dump(regr, 'reg.joblib') 