import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import os
from ent_purification import m1

file1 = os.path.join("data/", "ent_purif_xy_solutions10_01_2021_08_20_47.txt")
file2 = os.path.join("data/" "ent_purif_xy_solutions10_01_2021_07_12_33.txt")
file3 = os.path.join("data/", "ent_purif_xy_solutions10_01_2021_02_10_54.txt")
names = ["general U", "Z * MS * Z", "CNOT"]

for i_f, f in enumerate([file1, file2, file3]):
    data = np.loadtxt(f)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x = data[:, -2]
    y = data[:, -1]
    z = data[:, -4]
    plt.title(f"Gate {names[i_f]}")
    ax.plot_trisurf(x, y, z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False, label=names[i_f])
    # ax_2 = fig.add_subplot(110, projection='3d')
    # ax_2.scatter(data[:, 5], data[:, 7], np.round(data[:, 8], 3))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel(r'C(x,y)')
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
