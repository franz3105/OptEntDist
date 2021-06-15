import numpy as np
import scipy
import os
import datetime
import matplotlib.pyplot as plt

from scipy.optimize import minimize, basinhopping
from ent_purification import cost, rho_xy_phi, rho_xy, m1, m2, m3

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
    _, infidelity, prob = cost(rho, res.x, c_ops)
    print(infidelity, prob)
    return res.x, res.fun, infidelity, prob


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
                    best_values, best_error, best_infidelity, best_prob = None, 1., 1., 0.
                    for b in range(n_guess):
                        opt_values, error, infidelity, prob = optimize(rho, m1)
                        if best_values is None or error < best_error:
                            best_values = opt_values
                            best_error = error
                            best_infidelity = infidelity
                            best_prob = prob

    return solutions


def simulate_rho_xy():
    solutions = []
    n_guess = 10
    n_samples = 2
    x_array = np.zeros(n_samples)
    y_array = np.zeros(n_samples)
    phi = np.random.randn()
    # x_array = np.linspace(0.0, 1., n_samples)
    # y_array = np.sqrt(1 - x_array ** 2)
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
    for idx in range(n_samples):
        x = x_array[idx]
        y = y_array[idx]
        counter = counter + 1
        print(counter)
        print(phi, x, y, x**2 + y**2)
        rho, x_out, y_out = rho_xy(x, y, phase=phi)
        best_values, best_error, best_infidelity, best_prob = None, 1., 1., 0.
        for b in range(n_guess):
            opt_values, error, infidelity, prob = optimize(rho, m1)
            if best_values is None or error < best_error:
                best_values = opt_values
                best_error = error
                best_infidelity = infidelity
                best_prob = prob

        solutions.append(np.append(best_values, np.array([best_error, best_infidelity, best_prob, x, y])))

    return solutions


if __name__ == "__main__":
    sol_array = np.vstack(simulate_rho_xy())
    if not os.path.exists("data"):
        os.mkdir("data")
    now = datetime.datetime.now()
    np.savetxt(os.path.join("data", "ent_purif_xy_solutions" + now.strftime("%m_%d_%Y_%H_%M_%S") + ".txt"), sol_array)
