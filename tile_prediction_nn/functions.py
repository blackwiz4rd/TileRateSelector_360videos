import numpy as np

def quaternions_to_euler(q):
	q0 = q[:,0]
	q1 = q[:,1]
	q2 = q[:,2]
	q3 = q[:,3]
	#roll
	angle = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2**(q1**2 + q2**2))
	#pitch
	spitch = 2*(q0*q2 - q3*q1)
	mask = np.fabs(spitch)>=np.ones(spitch.size)
	pitch = np.array([])
	for i in range(mask.size):
		if mask[i] == True:
			pitch = np.append(pitch,np.copysign(np.pi,spitch[i]))
		else:
			pitch = np.append(pitch,np.arcsin(spitch[i]))
	angle = np.append(angle, pitch)
	#yaw
	angle = np.append(angle, np.arctan2(2*(q0*q3 + q1*q2), 1 - 2**(q2**2 + q3**2)))
	return angle.reshape(3,q.shape[0]).T

def fov_(x, y):
	margin_lateral = np.deg2rad(45)
	# margin_ab = np.deg2rad(90)
	margin_ab = np.deg2rad(110)
	fov = np.array([
		[x-margin_ab, y+margin_lateral], 
		[x+margin_ab, y+margin_lateral], 
		[x+margin_ab, y-margin_lateral], 
		[x-margin_ab, y-margin_lateral]
	])
	return fov

def rotation_matrix(w, x, y, z):
	n = w*w+x*x+y*y+z*z
	s = 0
	if n != 0:
		s = 2/n

	wx = s*w*x
	wy = s*w*y
	wz = s*w*z

	xx = s*x*x
	yy = s*y*y
	zz = s*z*z

	xy = s*x*y
	yz = s*y*z
	xz = s*x*z

	return np.array([
		[1 - (yy+zz), xy-wz, xz+wy],
		[xy+wz, 1-(xx+zz), yz-wx],
		[xz-wy, yz+wx, 1-(xx+yy)]
	])

# def rotation_2D(angle):
# 	c = np.cos(angle)
# 	s = np.sin(angle)
# 	return np.array([
# 		[c, -s],
# 		[s, c]
# 	])

def rotation_matrix_x(angle):
	c = np.cos(angle)
	s = np.sin(angle)
	return np.array([
		[1, 0, 0],
		[0, c, -s],
		[0, s, c]
	])

def rotation_matrix_y(angle):
	c = np.cos(angle)
	s = np.sin(angle)
	return np.array([
		[c, 0, s],
		[0, 1, 0],
		[-s, 0, c]
	])

def rotation_matrix_z(angle):
	c = np.cos(angle)
	s = np.sin(angle)
	return np.array([
		[c, -s, 0],
		[s, c, 0],
		[0, 0, 1]
	])

def rotated_vector_(yaw, pitch, roll, v):
	return np.matmul(rotation_matrix_z(yaw), np.matmul(rotation_matrix_y(pitch), np.matmul(rotation_matrix_x(roll),v)))

# equirectangular equations
def coordinates(fov, width, height):
	# x = (fov[:,0] + 180)/360 * width
	# y = (90 - fov[:,1])/180 * height
	x = (fov[:,0] + 180)/360 * width
	y = (90 - fov[:,1])/180 * height
	return np.array([x, y]).reshape(2,x.size).T

# check to which tile fov_top_left belongs
def check_row(y, height, n):
	h = height/n

	# base case
	if y < h:
		return n
	elif y >= (n-1)*h:
		return 1

	for i in np.arange(n-2)+1:#-2 cases
		if h*i <= y < h*(i+1):
			return n-i # n-(i+1)-1

def check_col(x, width, m):
	w = width/m
	if x < w:
		return 1
	elif x >= (m-1)*w:
		return m

	for i in np.arange(m-2)+1:#-2 cases
		if w*i <= x < w*(i+1):
			return i+1
