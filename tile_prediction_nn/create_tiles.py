import numpy as np
from functions import *
import os

#constants
# 4k
width = 4096
height = 2048
# 8k
# width = 7680
# height = 4320
# number of tiles n*m
n = m = 4

for userid in os.listdir('results'):
	if userid[0]!=".":
		filedir = '../results/'+userid+'/test0/Diving-2OzlksZBTiA/'
		# filedir = 'results/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Diving-2OzlksZBTiA/'
		if os.path.isfile(filedir+'Diving-2OzlksZBTiA_0.txt'):
			data = np.loadtxt(filedir+'Diving-2OzlksZBTiA_0.txt')

			q = data[:,2:6]
			c = data[:,1]

			#calculate roll pitch yaw
			angle = quaternions_to_euler(q)
			roll = angle[:,0]
			pitch = angle[:,1]
			yaw = angle[:,2]

			# WE WANT PITCH BETWEEN -90 AND 90, YAW BETWEEN -180 AND 180
			#trick to visualize data
			yaw[yaw < 0] = yaw[yaw < 0] + 2*np.pi
			yaw = yaw - np.pi

			# calculate 4 point which identify the fov based on yaw and pitch
			fov = np.array([])
			for i in range(angle.shape[0]):
				fov = np.append(fov, fov_(yaw[i], pitch[i]))
			fov = fov.reshape(angle.shape[0]*4,2)

			# radiant do degrees
			roll = np.rad2deg(roll)
			pitch = np.rad2deg(pitch)
			yaw = np.rad2deg(yaw)
			fov = np.rad2deg(fov)

			# split fov points in 4 subsets corresponding to the region they are belonging to
			# fov_top_left = fov[::4,:]
			fov_top_right = fov[1::4,:]
			# fov_bottom_right = fov[2::4,:]
			fov_bottom_left = fov[3::4,:]

			# calculate coordinates of fov
			# top_left = coordinates(fov_top_left, width, height)
			top_right = coordinates(fov_top_right, width, height)
			# bottom_right = coordinates(fov_bottom_right, width, height)
			bottom_left = coordinates(fov_bottom_left, width, height)

			# calculate tiles probabilities
			total_prob = np.array([])
			for i in range(bottom_left.shape[0]):
				tiles = np.arange(n*m)+1
				tiles = tiles.reshape(n,m)

				start_row = check_row(bottom_left[i,1], height,n)
				start_col = check_col(bottom_left[i,0], width,m)
				# print(tiles[start_row-1, start_col-1])

				end_row = check_row(top_right[i,1], height,n)
				end_col = check_col(top_right[i,0], width,m)
				# print(tiles[end_row-1, end_col-1])

				# print('fov', tiles[start_row-1:end_row, start_col-1:end_col])
				prob = np.copy(tiles)
				prob[start_row-1:end_row, start_col-1:end_col] = 0
				prob = prob==0
				prob = prob.reshape(n*m)
				prob = prob.astype(int)
				total_prob = np.append(total_prob, prob)

			total_prob = total_prob.reshape(bottom_left.shape[0],n*m)
			print(total_prob)
			np.save(filedir+'tiles', total_prob)