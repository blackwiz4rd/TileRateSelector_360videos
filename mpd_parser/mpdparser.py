import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def calculate_action(x, bitrate):
	# print(bitrate[bitrate.size-1], range(bitrate.size-2))
	if x >= bitrate[0]: # if bandwidth is higher than 5800 set 5800
		return 0
	elif x < bitrate[5]:
	 	return 6
	for i in range(bitrate.size-2):
		# print(bitrate[i + 1], bitrate[i])
		if x > bitrate[i + 1] and x < bitrate[i]:
			return i + 1
	

# PARSING MPD
# bitrate =  np.array([5800, 4300, 3750, 3000, 2350, 1750, 110]) # 7, from 0 to 6
# print(bitrate[calculate_action(100,bitrate)])
# print(bitrate[calculate_action(1000,bitrate)])
# print(bitrate[calculate_action(2000,bitrate)])
# print(bitrate[calculate_action(3400,bitrate)])
# print(bitrate[calculate_action(4000,bitrate)])
# print(bitrate[calculate_action(5000,bitrate)])
# exit()

tree = ET.parse('bigbuckbunny-simple.mpd')
root = tree.getroot()
data = np.array([])

n_segments = 0

for segment in root.iter('Segment'):
	n_segments = n_segments + 1
	for chunk in segment:
		data = np.append(data,chunk.attrib['quality'])
		representation = re.findall("\d+.\d+|\d", chunk.attrib['representation'])
		data = np.append(data,representation)

data = data.reshape(n_segments,8*2)

df = pd.DataFrame(data)
df = df.drop(np.arange(1,17,2), axis=1) # drop exceeding cols
# FINISHED PARSING MPD
df = pd.concat([df]*38).reset_index(drop=True)
print(df.shape)

bandwidth = np.load('../bandwidth_extractor/bandwidth.npy')/1e3 # retreiving bandwidth
bandwidth = np.tile(bandwidth,38)
print(bandwidth.shape)
prob = np.load('../tile_prediction_nn/prob.npy') # estimated nn probability
true_prob = np.load('../tile_prediction_nn/y_validation.npy') # true probability
# print(true_prob)
print(prob.shape, true_prob.shape)
# adding columns to dataframe
df['bandwidth'] = bandwidth # in kbit/s
df['prob'] = prob.reshape(prob.shape[0]*prob.shape[1])[:bandwidth.size]
df['true'] = true_prob.reshape(prob.shape[0]*prob.shape[1])[:bandwidth.size]

df = df.iloc[:-2] # drop last two rows

temp = np.add.reduceat(df['bandwidth'], range(0, df['bandwidth'].size, 16))
temp_bandwidth16 = temp
df['bandwidth16'] = np.repeat(temp,16)
print(df['bandwidth16'].shape)

temp = np.add.reduceat(df['prob'], range(0, df['prob'].size, 16))
df['bandw'] = df['bandwidth16']*df['prob']/np.repeat(temp,16)

temp = np.add.reduceat(df['true'], range(0, df['true'].size, 16))
df['perfect_bandw'] = df['bandwidth16']*df['true']/np.repeat(temp,16)

bitrate =  np.array([5800, 4300, 3750, 3000, 2350, 1750, 110]) # 7, from 0 to 6
action = np.array([])
perfect_action = np.array([])
df = df.apply(pd.to_numeric) # convert data to numeric
# print(df['bandw'].size)
for i in range(df['bandw'].size):
	x = df['bandw'][i]
	y = df['perfect_bandw'][i]
	action = np.append(action, bitrate[calculate_action(x, bitrate)])
	perfect_action = np.append(perfect_action, bitrate[calculate_action(y, bitrate)])
# calculate action per tile
# print(action.shape)
# print(perfect_action.shape)
df['action'] = action
# print(df['action'][144],df['action'][145])
# for x in bitrate:
	# print(x, action.count(x))
	# print(x, perfect_action.tolist().count(x))

df.columns = ['action','perfect_bandw', 'bandw', 'bandwidth16', 'true', 'prob', 'bandwidth', '750k', '5800k', '4300k', '3750k', '3000k', '2350k', '1750k', '110k'][::-1]
# 375k -> 3750k

# calculate maximum achieved quality
achieved_quality = np.array([])
best_quality = np.array([])
for i in range(action.size):
	achieved_quality = np.append(achieved_quality, df[str(int(action[i]))+'k'][i])
	best_quality = np.append(best_quality, df[str(int(perfect_action[i]))+'k'][i])

def delta_qt(qt, prob):
	delta_qt = qt*prob
	delta_qt = delta_qt.reshape(int(qt.size/16), 16).sum(axis=1)
	temp = delta_qt[1:]
	temp = np.append(temp, delta_qt[delta_qt.size-1])
	return delta_qt - temp

achieved_delta_quality = delta_qt(achieved_quality,np.array(df["true"]))
best_delta_quality = delta_qt(best_quality,np.array(df["true"]))
# show results
print('maximum overall achievable quality: ', np.sum(df['5800k']))
print('maximum achieved quality with predicted tiles:', np.sum(achieved_quality*df["true"]))
print('maximum achievable quality with true tiles: ', np.sum(best_quality*df["true"]))
print('-------')
print('maximum achieved delta quality with predicted tiles:', np.sum(np.abs(achieved_delta_quality)))
print('maximum achievable delta quality with true tiles:', np.sum(np.abs(best_delta_quality)))
# print(df)
print(df.iloc[:,:7])

fig1, ax1 = plt.subplots()
# in viewport
achieved_quality = np.array(achieved_quality*df["true"])[np.nonzero(achieved_quality*df["true"])]
best_quality = np.array(best_quality*df["true"])[np.nonzero(best_quality*df["true"])]
ax1.boxplot([achieved_quality, best_quality])
plt.xticks([1, 2], ["Achieved quality", "Best quality achievable"])
plt.ylabel('Quality (SSIM)')
plt.show()

fig1, ax1 = plt.subplots()
# in viewport
ax1.boxplot([achieved_delta_quality, best_delta_quality])
print(np.sum(np.abs(achieved_delta_quality)))
print(np.sum(np.abs(best_delta_quality)))
plt.xticks([1, 2], ["Achieved delta quality in viewport","Best delta quality in viewport"])
plt.ylabel('Delta quality (SSIM)')
plt.show()

## plot capacity in kbit/s from given bit/s
ax = plt.figure().gca()
ax.scatter(np.arange(temp_bandwidth16.size), temp_bandwidth16, c='darkslateblue')
plt.plot(np.arange(temp_bandwidth16.size), temp_bandwidth16, 'darkslateblue')
ax.set_xlabel('segment')
ax.set_ylabel('bandwidth (kbit/s)')
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()