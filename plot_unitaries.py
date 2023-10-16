import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import zeros
import os

def ket_str(n_qudits, d=2, bra_ket = 1):
    str_list = []
    #how_many = d**n_qudits
    nums = np.empty([d,1], dtype=int)
    nums[:,0] = np.arange(d)
    for i in range(n_qudits-1):
        prev_nums = nums.copy()
        s = prev_nums.shape
        nums = np.empty([d*s[0], s[1]+1])
        for j in range(d):
            nums[j*s[0]:(j+1)*s[0], 1:] = prev_nums
            nums[j*s[0]:(j+1)*s[0],0] = j
    for i in range(nums.shape[0]):
        if bra_ket:
            new_str = r'$\ket{'
        else:
            new_str = r'$\bra{'
        for j in range(nums.shape[1]):
            new_str += str(int(nums[i,j]))
            #if j < nums.shape[1]-1
            #    new_str += ','
        new_str += r'}$'
        str_list.append(new_str)
    return str_list



def unitary_plot_separate(U, idx, cmap='Oranges', ax=[], fig=[], phase_mode=0):
    plt.rcParams["figure.figsize"] = [7, 6]
    plt.rcParams["figure.autolayout"] = True

    SMALL_SIZE = 15
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 15

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # Maybe choose a repeating colormap like hsv

    s = U.shape[0]
    s2 = U.shape[1]
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cmap_gray = plt.get_cmap('gray_r')
    cmat = cmap(range(256))
    U2 = zeros((s, 1))
    U2[0] = -1
    U2[1] = 1
    if phase_mode == 0:
        R = np.real(U)
        I = np.imag(U)
        title = ['Real', 'Imag']
    else:
        R = np.abs(U)
        print(R)
        phi = np.angle(U)
        phi = phi + np.pi
        I = np.mod(phi, np.pi) / np.pi
        # print(I)
        title = ['$|U_{ij}|$', 'arg$(U_{ij})$']
    if isinstance(ax, list):
        fig, ax = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [15, 15, 1]})
    ax[0].imshow(R, cmap=cmap, vmin=0, vmax=1)
    ax[0].set_xticklabels(["0", "1", "2", "3", "4"])
    ax[0].set_yticklabels(["0", "1", "2", "3", "4"])
    ax[0].set_xlabel("column number", size=20)
    ax[0].set_ylabel("row number", size=20)
    ax[0].set_title(title[0], fontsize=30)
    ax[0].tick_params(axis='both', which='major', labelsize=24)
    ax[1].imshow(I, cmap=cmap, vmin=0, vmax=1)
    ax[1].set_xticklabels(["0", "1", "2", "3", "4"])
    ax[1].set_yticklabels(["0", "1", "2", "3", "4"])
    ax[1].set_xlabel("column number", size=20)
    ax[1].set_ylabel("row number", size=20)
    ax[1].set_title(title[1], fontsize=30)
    ax[1].tick_params(axis='both', which='major', labelsize=24)
    for i in range(2):
        ax[i].set(xlim=[-0.5, s - 0.5], ylim=[-0.5, s2 - 0.5])
        ax[i].invert_yaxis()
    norm = colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    caxer = fig.colorbar(sm, ticks=[0, 0.5, 1], label='Values', cax=ax[2])
    caxer.ax.tick_params(labelsize=20)
    caxer.set_label(label='Numerical value', size=25)
    # cax.ax.set(yticks=[0,0.5,1])


def plot():
    U = np.loadtxt(os.path.join("data", "XY_avg_generalU", 'best_unitaries_3.txt'), dtype=np.complex_).reshape(6,
                                                                                                           4,
                                                                                                           4)
    # U = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])
    for i in range(6):
        unitary_plot_separate(U[i, ::], i, phase_mode=1)
        plt.savefig(os.path.join("plots", f'best_unitaries_genU_plot_{i}.pdf'), dpi=300,
                    format='pdf',
                    bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    plot()
