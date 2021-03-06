#load the required packages
from joblib import dump, load
import numpy as np
import os
import sklearn
from sklearn.neural_network import MLPClassifier
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
clf = MLPClassifier(hidden_layer_sizes=(512), verbose=True, max_iter=300, alpha=1e-4, solver='sgd', tol=1e-4, random_state=ID, learning_rate_init=.01)
# clf = MLPClassifier(hidden_layer_sizes=(512), verbose=True, max_iter=100, alpha=1e-4, solver='sgd', tol=1e-4, random_state=ID, learning_rate_init=.1)
clf.fit(X_train, y_train)
# print(clf.out_activation_)

# print(clf.predict_proba(X_test))
training_error = 1. - clf.score(X_train,y_train) #ADD YOUR CODE
test_error = 1. - clf.score(X_test,y_test) #ADD YOUR CODE

# print(clf.predict(X_test))

# print ('\nRESULTS FOR BEST NN\n')

print ("Best NN training error: %f" % training_error)
print ("Best NN test error: %f" % test_error)

# save the trained model
dump(clf, 'nn.joblib') 
