import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# data format
format0 = ['capacity', 'action', 'quality', 'buffer', 'rebuffering_time', 'reward', 'state1', 'state2', 'state3', 'state4', 'state5', 'state6', 'state7', 'state8']

# preprocessing on data
with open('MPD4log', 'r') as f:
    content = f.readlines()[3:]

content = [x.strip() for x in content]
content = np.array(content)
content = content[::2]
data = np.array([])
for x in content:
	data = np.append(data,re.findall("\d+.\d+|\d", x))

data = np.array([float(x) for x in data])
data = data.reshape(content.shape[0],len(format0))

# dataframe
df = pd.DataFrame(data)
df.columns = format0
print(df)
# print(df.iloc[147,:])
# print(df.iloc[0,:],'\n')
# df = df.iloc[::2]
# print(df.iloc[1,:])
# print(df)
# print(df.shape)
# print(df[df['action']==6])

## plot capacity in kbit/s from given bit/s
fig, ax = plt.subplots()
# p = ax.scatter(np.arange(df.shape[0]), df['capacity']/1e3, c=df['action'])
p = ax.scatter(np.arange(df.shape[0]), df['capacity']/1e3)
plt.plot(np.arange(df.shape[0]), df['capacity']/1e3)
# plt.colorbar(p)
ax.set_xlabel('tile #')
ax.set_ylabel('bandwidth [kbit/s]')

np.save('bandwidth', df['capacity'])
plt.show()