import numpy as np

viewport = [0.2, 0.2, 0.2, 0.2]
outside = [0.1, 0.05, 0.05]
offline = [0.00, 0.00]

p1 = viewport + outside + offline
p2 = outside + viewport + offline
p3 = offline + viewport + outside
p4 = offline + outside + viewport

p = np.array([p1, p2, p3, p4])
p = p.reshape(4,9)
# print(p)

prob = np.array([])
for j in range(int(300/4)):
	for i in range(p.shape[0]):
		prob = np.append(prob, np.random.choice(9, 9, p=p[i])+1)
		prob = prob/np.sum(prob)

prob = prob.reshape(300,9)
print(prob)