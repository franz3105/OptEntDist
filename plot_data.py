import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from ent_purification import m1

file = os.path.join("data", "ent_purif_xy_solutions05_31_2021_12_51_10.txt")
data = np.loadtxt(file)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print(data[:, 8])
ax.scatter(data[:, 9], data[:, 10], np.round(data[:, 11], 3))
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel(r'1 - F')
plt.show()

for s in data[:, :8]:
    print(s)
m_array = [m1(s) for s in data[:, :8]]
x_array = data[:, 5]
y_array = data[:, 6]
print(x_array)

dist_plot = np.zeros(len(x_array))
for i_m, m in enumerate(m_array):
    dist = 0
    for index in [-3, -2, -1, 1, 2, 3]:
        try:
            dist += np.linalg.norm(m_array[i_m] - m_array[i_m + index])
        except:
            pass
        dist_plot[i_m] = dist

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(data[:, 5], data[:, 6], dist_plot)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Average Frobenius distance')
#plt.show()
