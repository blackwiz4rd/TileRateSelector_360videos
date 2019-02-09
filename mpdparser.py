import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def calculate_action(x, bitrate):
	if x >= bitrate[0]: # if bandwidth is higher than 5800 set 5800
		return 0
	if x <= bitrate[bitrate.size-1]: # if bandwidth is lower than 1750 set 110
		return bitrate.size-1
	for i in range(bitrate.size-2):
		if x > bitrate[i + 1] and x < bitrate[i]:
			return i + 1

# PARSING MPD
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

bandwidth = np.load('bandwidth.npy')/1e3 # retreiving bandwidth
prob = np.load('prob.npy') # estimated nn probability
true_prob = np.load('results/uid-9f0512a7-3258-40bd-bd03-6d783bfb1e99/test0/Diving-2OzlksZBTiA/tiles.npy')[1:] # true probability
print(prob.shape, true_prob.shape)
# adding columns to dataframe
df['bandwidth'] = bandwidth # in kbit/s
df['prob'] = prob.reshape(prob.shape[0]*prob.shape[1])[200:200+bandwidth.size]
df['true'] = true_prob.reshape(prob.shape[0]*prob.shape[1])[200:200+bandwidth.size]

df.drop(df.tail(11).index,inplace=True) # drop last 11 rows
temp = np.add.reduceat(df['bandwidth'], range(0, df['bandwidth'].size, 16))
df['bandwidth16'] = np.repeat(temp,16)
print(df['bandwidth16'].shape)

temp = np.add.reduceat(df['prob'], range(0, df['prob'].size, 16))
df['bandw'] = df['bandwidth16']*df['prob']/np.repeat(temp,16)

temp = np.add.reduceat(df['true'], range(0, df['true'].size, 16))
df['perfect_bandw'] = df['bandwidth16']*df['true']/np.repeat(temp,16)

bitrate =  np.array([5800, 4300, 3750, 3000, 2350, 1750, 110]) # 7, from 0 to 6
action = np.array([])
perfect_action = np.array([])
for i in range(df['bandw'].size):
	x = df['bandw'][i]
	y = df['perfect_bandw'][i]
	action = np.append(action, bitrate[calculate_action(x, bitrate)])
	perfect_action = np.append(perfect_action, bitrate[calculate_action(y, bitrate)])

# calculate action per tile
df['action'] = action
for x in bitrate:
	print(x, action.tolist().count(x))
	print(x, perfect_action.tolist().count(x))

df.columns = ['action','perfect_bandw', 'bandw', 'bandwidth16', 'true', 'prob', 'bandwidth', '750k', '5800k', '4300k', '3750k', '3000k', '2350k', '1750k', '110k'][::-1]
# 375k -> 3750k
df = df.apply(pd.to_numeric) # convert data to numeric

# calculate maximum achieved quality
achieved_quality = np.array([])
best_quality = np.array([])
for i in range(action.size):
	# achieved_quality = np.append(achieved_quality, df[str(int(action[i]))+'k'][i]*df['true'][i])
	# best_quality = np.append(best_quality, df[str(int(perfect_action[i]))+'k'][i]*df['true'][i])
	achieved_quality = np.append(achieved_quality, df[str(int(action[i]))+'k'][i])
	best_quality = np.append(best_quality, df[str(int(perfect_action[i]))+'k'][i])

# show results
print('maximum overall achievable quality: ', np.sum(df['5800k']))
print('maximum achieved quality with predicted tiles:', np.sum(achieved_quality))
print('maximum achievable quality with true tiles: ', np.sum(best_quality))
# print(df)
# print(df.iloc[:,:7])

fig1, ax1 = plt.subplots()
ax1.boxplot([achieved_quality, best_quality], showfliers=True)
plt.xticks([1, 2], ["Achieved quality", "Best quality achievable"])
plt.show()

# plt.imshow(true_prob[200:300,:].T,interpolation='none',aspect='auto')
# plt.xlabel('i-th tile')
# plt.ylabel('j-th segment')
# plt.colorbar()
# plt.yticks(np.arange(1,16))
# plt.show()
