import numpy as np
import scipy
import os
import datetime
import matplotlib.pyplot as plt

from scipy.optimize import minimize, basinhopping
from ent_purification import cost, rho_xy_phi, rho_xy, m1, m2

counter = 0


def optimize(rho, c_ops):
    function = lambda x: cost(rho, x, c_ops)[0]
    dim = 4
    init_values = np.random.randn(dim)
    history = list()
    counter = 0

    def store(x):
        history.append(cost(rho, x, c_ops)[1:])

    minimizer_kwargs = {"method": "BFGS"}

    # res = basinhopping(function, init_values, minimizer_kwargs=minimizer_kwargs,
    #                   niter=30, disp=True)
    res = minimize(function, x0=init_values, method="l-bfgs-b", callback=store)
    print(res.fun)
    # print(res.x)
    return res.x, res.fun, history


def simulate_rho_xy_phi():
    solutions = []
    n_guess = 3
    phi_array = np.linspace(0.1, np.pi, 40)
    eta_damp_array = np.linspace(0.1, 0.5, 1)
    n_photon_array = np.linspace(0, 1000, 10)
    for phi in phi_array:
        for eta in eta_damp_array:
            for n in n_photon_array:
                rho, x, y = rho_xy_phi(eta, phi, n)
                for b in range(n_guess):
                    opt_values, error, history = optimize(rho, m1)
                    print(history[-1])
                    solutions.append(np.append(opt_values, np.array([error, x, y, history[-1][0], history[-1][0]])))

    return solutions


def simulate_rho_xy():
    solutions = []
    n_guess = 10
    n_samples = 1000
    x_array = np.zeros(n_samples)
    y_array = np.zeros(n_samples)
    phi_array = np.linspace(0.5, 2 * np.pi - 0.1, 100)

    for i_rand in range(n_samples):
        t = 2 * np.pi * np.random.random()
        u = np.random.random() + np.random.random()
        r = 2 - u if u > 1 else u
        x, y = r * np.cos(t), r * np.sin(t)
        x_array[i_rand] = x
        y_array[i_rand] = y

    plt.scatter(x_array, y_array)
    plt.show()
    counter = 0
    for phi in phi_array:
        for idx in range(n_samples):
            x = x_array[idx]
            y = y_array[idx]
            counter = counter + 1
            print(counter)
            print(phi, x, y)
            rho, x_out, y_out = rho_xy(x, y, phase=phi)
            best_values, best_error, best_history = None, 1., None
            for b in range(n_guess):
                opt_values, error, history = optimize(rho, m1)
                if best_values is None:
                    best_values = opt_values
                    best_history = history
                if error < best_error:
                    best_error = error
                    best_values = opt_values
                    best_history = history
            print(best_history)
            solutions.append(np.append(best_values, np.array([best_error, x, y]),
                                       np.array([best_history[0], best_history[1]])))

    return solutions


sol_array = np.vstack(simulate_rho_xy())
if not os.path.exists("data"):
    os.mkdir("data")
now = datetime.datetime.now()
np.savetxt(os.path.join("data", "ent_purif_xy_solutions" + now.strftime("%m_%d_%Y_%H_%M_%S") + ".txt"), sol_array)
