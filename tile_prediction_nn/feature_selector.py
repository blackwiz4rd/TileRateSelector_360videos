import numpy as np

def feature_selector_training(quaternions, tiles, timestamps):
	# for each quaternion find the one at 2s distance
	# print(quaternions.shape, tiles.shape, timestamps.shape)
	temp_quaternions = np.array([])
	temp_tiles = np.array([])
	index = []
	for i in range(timestamps.size):
		j = i + 1
		try:
			# 0.5 = near future prediction in seconds
			# 1 = mid future prediction in seconds
			# 2 = near future prediction in seconds
			while timestamps[j]-timestamps[i] < 0.5:
				j = j + 1
			index.append(j)
		except IndexError:
			pass
		
	# print(timestamps[0:10], timestamps[index[0:10]])
	# print(len(index))

	for i in range(len(index)):
		temp_quaternions = np.append(temp_quaternions, quaternions[i:i+5,:])
		temp_tiles = np.append(temp_tiles, tiles[index[i]])

	return temp_quaternions, temp_tiles

# def feature_selector(quaternions, tiles, timestamps):
# 	steps = np.arange(0,72,1)
# 	index = []
# 	i = 0
# 	for t in range(timestamps.size):
# 		if(int(timestamps[t]) == steps[i]):
# 			index.append(t)
# 			i = i + 1
# 		if(steps[i] == 70):
# 			break

# 	# print('valid indexes, length of valid indexes', index, len(index))

# 	temp_quaternions = np.array([])
# 	temp_tiles = np.array([])
# 	for i in range(len(index)-1):
# 		temp_quaternions = np.append(temp_quaternions, quaternions[index[i]:index[i]+5,:])
# 		temp_tiles = np.append(temp_tiles, tiles[index[i+1]])

# 	# print("temp_quaternions, temp_tiles shape", temp_quaternions.shape, temp_tiles.shape)
# 	return temp_quaternions, temp_tiles