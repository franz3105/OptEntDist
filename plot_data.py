import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import os
from ent_purification import m1

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

file = os.path.join("data", "ent_purif_xy_solutions06_02_2021_17_38_04.txt")
data = np.loadtxt(file)
x = data[:, 7]
y = data[:, 8]
z = data[:, 6]
print(x.shape)
print(y.shape)
print(z.shape)
# x,y = np.meshgrid(x,y)
ax.plot_trisurf(x, y, z, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
# ax_2 = fig.add_subplot(110, projection='3d')
# ax_2.scatter(data[:, 5], data[:, 7], np.round(data[:, 8], 3))
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel(r'1 - p')
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
# plt.show()
