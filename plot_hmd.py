import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions import *

# data = np.loadtxt('results/uid-00d6d7f2-23df-4062-84dd-d5e99183dae1/test0/Diving-2OzlksZBTiA/Diving-2OzlksZBTiA_0.txt')
data = np.loadtxt('results/uid-9f0512a7-3258-40bd-bd03-6d783bfb1e99/test0/Diving-2OzlksZBTiA/Diving-2OzlksZBTiA_0.txt')

q = data[:,2:6]
c = data[:,1]

#roll pitch yaw
angle = quaternions_to_euler(q)
roll = angle[:,0]
pitch = angle[:,1]
yaw = angle[:,2]

# # 3D PLOT WITH ROTATION MATRIX

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# unit_vector = np.array([1, 1, 1])
# unit_vector = unit_vector.reshape(3,1)
# rotated_vector = np.array([])

# for i in range(angle.shape[0]):
# 	rotated_vector = np.append(rotated_vector, rotated_vector_(roll[i], pitch[i], yaw[i], unit_vector))

# rotated_vector = rotated_vector.reshape(data.shape[0], 3)

# p = ax.scatter(rotated_vector[:,0], rotated_vector[:,1], rotated_vector[:,2], c=c)
# fig.colorbar(p)
# ax.set_title('3D PLOT WITH ROTATION MATRIX from Euler angles')
# plt.show()

# # 3D PLOT WITH ROTATION MATRIX FROM QUATERNIONS

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# rotated_vector = np.array([])

# for i in range(angle.shape[0]):
# 	rotated_vector = np.append(rotated_vector, np.matmul(rotation_matrix(q[i,0], q[i,1], q[i,2], q[i,3]), unit_vector))

# rotated_vector = rotated_vector.reshape(data.shape[0], 3)

# p = ax.scatter(rotated_vector[:,0], rotated_vector[:,1], rotated_vector[:,2], c=c)
# ax.set_title('3D PLOT WITH ROTATION MATRIX from quaternions')
# fig.colorbar(p)
# plt.show()

# # 2D PLOT WITH EULER ANGLES
yaw[yaw < 0] = yaw[yaw < 0] + 2*np.pi #trick to visualize data
# SAME AS DATASET PAPER
# yaw = yaw - np.pi
# pitch = pitch - np.pi/2
# WE WANT PITCH BETWEEN -90 AND 90, YAW BETWEEN -180 AND 180
yaw = yaw - np.pi

fov = np.array([])
for i in range(angle.shape[0]):
	fov = np.append(fov, fov_(yaw[i], pitch[i]))

fov = fov.reshape(angle.shape[0]*4,2)

# angle = np.rad2deg(angle)
fig, ax = plt.subplots(figsize=(8,6))
p = ax.scatter(yaw, pitch,c=c)
plt.colorbar(p)
ax.set_yticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
ax.set_yticklabels([r"$-\frac{1}{2}\pi$", r"$-\frac{1}{4}\pi$", r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$"])

ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$-\frac{1}{2}\pi$", r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$"])
ax.set_ylabel('latitude (radian)')
ax.set_xlabel('longitude (radian)')
plt.show()

roll = np.rad2deg(roll)
pitch = np.rad2deg(pitch)
yaw = np.rad2deg(yaw)
fov = np.rad2deg(fov)
plt.scatter(yaw, pitch,c=c)
# fov plot
plt.plot(fov[:,0], fov[:,1], '.')
# plt.colorbar(p)
plt.title('2D PLOT attempt from angles')
plt.show()

# 2D PLOT IN VIDEO RESOLUTION
# 4k
width = 4096
height = 2048
# 8k
# width = 7680
# height = 4320

fov_top_left = fov[::4,:]
fov_top_right = fov[1::4,:]
fov_bottom_right = fov[2::4,:]
fov_bottom_left = fov[3::4,:]

top_left = coordinates(fov_top_left, width, height)
top_right = coordinates(fov_top_right, width, height)
bottom_right = coordinates(fov_bottom_right, width, height)
bottom_left = coordinates(fov_bottom_left, width, height)


n = m = 4
tiles = np.arange(n*m)+1
tiles = tiles.reshape(n,m)
print(tiles)
start_row = check_row(bottom_left[0,1], height,n)
start_col = check_col(bottom_left[0,0], width,m)
print(tiles[start_row-1, start_col-1])

end_row = check_row(top_right[0,1], height,n)
end_col = check_col(top_right[0,0], width,m)
print(tiles[end_row-1, end_col-1])

print('fov', tiles[start_row-1:end_row, start_col-1:end_col])
prob = np.copy(tiles)
prob[start_row-1:end_row, start_col-1:end_col] = 0
prob = prob==0
prob = prob.reshape(n*m)
prob = prob.astype(int)
print(prob)

h = height/n
w = width/m
plt.axhline(h)
plt.axhline(2*h)
plt.axhline(3*h)
plt.axvline(w)
plt.axvline(2*w)
plt.axvline(3*w)
plt.axvline(width, color='g')
plt.axhline(height, color='g')
plt.axvline(0, color='g')
plt.axhline(0, color='g')

plt.plot(top_left[:,0],top_left[:,1],'c.')
plt.plot(top_right[:,0],top_right[:,1],'c.')
plt.plot(bottom_left[:,0],bottom_left[:,1],'c.')
plt.plot(bottom_right[:,0],bottom_right[:,1],'c.')

line_x = [bottom_left[0,0], bottom_right[0,0], top_right[0,0], top_left[0,0], bottom_left[0,0]]
line_y = [bottom_left[0,1], bottom_right[0,1], top_right[0,1], top_left[0,1], bottom_left[0,1]]
plt.plot(line_x, line_y, 'k')
plt.plot(bottom_left[0,0], bottom_left[0,1], 'kx')
plt.plot(top_right[0,0], top_right[0,1], 'kx')
plt.plot(bottom_right[0,0], bottom_right[0,1], 'kx')
plt.plot(top_left[0,0], top_left[0,1], 'kx')
plt.xlabel("width [pixel]")
plt.ylabel("height [pixel]")
plt.xlim([0,width])
plt.ylim([0,height])
# plt.title('2D PLOT on video')
plt.show()

# GROUP DATA BY TILES
