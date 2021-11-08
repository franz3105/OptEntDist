import numpy as np
import scipy
import os
import datetime
import matplotlib.pyplot as plt
import qutip as qt

from scipy.optimize import minimize
from ent_purification import cost, m1
from qutip_wrapper import rho_xy, rho_xy_phi, rho_zsolt, werner, pr01
import multiprocessing as mp
import jax.numpy as jnp
from jax import grad, jit
from jax_minimize_wrapper import minimize

counter = 0


def optimize(rho, n_iter, init_values):
    # print(rho)

    def opt_func(x):
        return cost(rho, x, n_iter - 1)[0]

    res = minimize(opt_func, x0=init_values, method="bfgs")
    cost_value, prob, *_ = cost(rho, res.x, n_iter - 1)

    return [res.fun, res.x, prob]


def simulate_rho_xy():
    solutions = []
    n_guess = 1
    n_samples = 10000
    n_iter = 1
    x_array = np.zeros(n_samples)
    y_array = np.zeros(n_samples)
    phi = 0
    # x_array = np.linspace(0.0, 1., n_samples)
    # y_array = np.sqrt(1 - x_array ** 2)
    for i_rand in range(n_samples):
        t = 2 * np.pi * np.random.random()
        u = np.random.random() + np.random.random()
        r = 2 - u if u > 1 else u
        x, y = r * np.cos(t), r * np.sin(t)
        x_array[i_rand] = x
        y_array[i_rand] = y

    cost_minima = []
    cost_args = []
    probs = []

    for idx in range(n_samples):
        x = x_array[idx]
        y = y_array[idx]
        print(idx, x ** 2 + y ** 2)
        rho, x_out, y_out = rho_xy(x, y, phase=phi)

        print(qt.concurrence(rho))
        dim = 15 * n_iter
        init_values = np.random.randn(dim)
        res = optimize(jnp.array(rho.full(), dtype=np.complex128), n_iter, init_values)

        cost_minima.append(res[0])
        cost_args.append(res[1])
        probs.append(res[2])

        best_idx = int(np.argmin(cost_minima))
        best_values = cost_args[best_idx]
        best_concurrence = 1 - np.min(cost_minima)
        print(f"Final value: {best_concurrence}")
        best_prob = probs[best_idx]
        # print(best_prob)
        # print(qt.concurrence(rho))

        solutions.append(np.append(best_values, np.array([best_concurrence, best_prob, x, y])))

    return solutions


def simulate_werner():
    solutions = []
    n_guess = 1
    n_samples = 100
    n_iter = 1

    c = np.linspace(2 / 3, 1, n_samples)
    Q = mp.Queue()

    ctr = 0
    for idx in range(n_samples):
        ctr = ctr + 1
        rho = werner(c[idx]).full()
        jobs = []
        print(1 - qt.concurrence(rho))
        for b in range(n_guess):
            dim = 15 * n_iter
            init_values = 2 * np.pi * np.random.uniform(0, 1, dim)
            p = mp.Process(target=optimize, args=(rho, m1, n_iter, Q, init_values))
            p.start()
            jobs.append(p)

        for p in jobs:
            p.join()

        results = [Q.get() for _ in range(n_guess)]
        cost_minima = np.array([r[0] for r in results])
        # print(cost_minima)
        cost_args = np.array([r[1] for r in results])
        probs = np.array([r[2] for r in results])

        best_idx = int(np.argmin(np.asarray(cost_minima)))
        best_values = cost_args[best_idx]
        best_concurrence = cost_minima[best_idx]
        print(best_concurrence)
        best_prob = probs[best_idx]
        # print(best_prob)

        solutions.append(np.append(best_values, np.array([best_concurrence, best_prob])))

    return solutions


def simulate_rho_c():
    solutions = []
    n_guess = 50
    n_samples = 1000
    n_iter = 1

    c = np.linspace(0.5, 1, n_samples)
    Q = mp.Queue()

    ctr = 0
    for idx in range(n_samples):
        ctr = ctr + 1
        rho = rho_zsolt(c[idx])
        jobs = []
        for b in range(n_guess):
            dim = 4 * n_iter
            init_values = 2 * np.pi * np.random.uniform(0, 1, dim)
            p = mp.Process(target=optimize, args=(rho, m1, n_iter, Q, init_values))
            p.start()
            jobs.append(p)

        for p in jobs:
            p.join()

        results = [Q.get() for _ in range(n_guess)]
        cost_minima = np.array([r[0] for r in results])
        # print(cost_minima)
        cost_args = np.array([r[1] for r in results])
        probs = np.array([r[2] for r in results])

        best_idx = int(np.argmin(np.asarray(cost_minima)))
        best_values = cost_args[best_idx]
        best_concurrence = cost_minima[best_idx]
        print(best_concurrence)
        best_prob = probs[best_idx]
        # print(best_prob)

        solutions.append(np.append(best_values, np.array([best_concurrence, best_prob])))

    return solutions


def main():
    sol_array = np.vstack(simulate_rho_xy())
    print(sol_array)
    if not os.path.exists("data"):
        os.mkdir("data")
    now = datetime.datetime.now()
    np.savetxt(os.path.join("data", "ent_purif_xy_solutions" + now.strftime("%m_%d_%Y_%H_%M_%S") + ".txt"), sol_array)


if __name__ == "__main__":
    main()
