#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 08:32:24 2021

@author: zsolt
"""
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import jax.numpy as jnp
import jax
import qutip as qt

from jax.scipy.linalg import expm
from scipy.optimize import minimize
from jax.ops import index, index_update
from jax import jit, grad
from functools import partial
from qutip_wrapper import werner, rho_zsolt
from jax_minimize_wrapper import minimize

jax.config.update('jax_enable_x64', True)

# import qutip as qt
# from numpy import random
p11 = jnp.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.complex128)
p10 = jnp.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.complex128)
p01 = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=jnp.complex128)
p00 = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=jnp.complex128)
Id4 = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=jnp.complex128)

X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

Id2 = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex128)

CNOT = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex128)


def get_generators():
    clifford_set = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    generators = []
    for sigma_1 in clifford_set:
        for sigma_2 in clifford_set:
            generators.append(jnp.array(qt.tensor(sigma_2, sigma_1).full()))

    del (generators[0])

    return generators


generators = get_generators()


def concurrence(rho: np.ndarray):
    YY = jnp.kron(Y, Y)
    rho_tilde = (YY.dot(rho.T)).dot(YY)
    eigs = -jnp.sort(-jnp.linalg.eigvals(rho.dot(rho_tilde)))
    comp_arr = jnp.array([0., jnp.sqrt(eigs[0]) - jnp.sum(jnp.sqrt(eigs[1:]))])

    return jnp.max(jnp.real(comp_arr))


def partial_trace(rho: np.ndarray, subscripts='ijklmnkl->ijmn'):
    rho_rs = jnp.reshape(rho, [2, 2, 2, 2, 2, 2, 2, 2])
    rho_pt = jnp.einsum(subscripts, rho_rs).reshape(4, 4)

    return rho_pt


def tensor(qubit_list):
    s = None

    for i_el, el in enumerate(qubit_list):
        if i_el == 0:
            s = el
        else:
            s = jnp.kron(s, el)

    return s


def general_U(s: np.ndarray):
    o = tensor([jnp.zeros((2, 2))] * 2)
    for i in range(s.shape[0]):
        o += s[i] * generators[i]

    return o


def apply_entangling_gate(rho: np.ndarray, gate: np.ndarray):
    # if a is not 3D vector, then this has to be changed
    G = gate

    rho_f = jnp.kron(rho, rho)
    U = jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=jnp.complex128)
    UU = jnp.kron(jnp.kron(Id2, U), Id2)

    # side A
    MA = jnp.kron(expm(-1j * G), Id4)
    MAc = jnp.kron(expm(1j * G), Id4)
    UA = (UU.dot(MA)).dot(UU)
    UAc = (UU.dot(MAc)).dot(UU)

    # side B
    MB = jnp.kron(Id4, expm(-1j * G))
    MBc = jnp.kron(Id4, expm(1j * G))
    UB = (UU.dot(MB)).dot(UU)
    UBc = (UU.dot(MBc)).dot(UU)

    rho_t = (((UA.dot(UB)).dot(rho_f)).dot(UBc)).dot(UAc)

    return rho_t


def purification(rho: np.ndarray, gate: np.ndarray):
    rho_t = apply_entangling_gate(rho, gate)

    proj_list = [p00, p01]
    c_arr = jnp.array([0, 0], dtype=jnp.complex128)
    p_arr = jnp.array([0, 0], dtype=jnp.complex128)
    rf_arr = jnp.zeros((2, rho.shape[0], rho.shape[1]), dtype=jnp.complex128)

    for i_p, p in enumerate(proj_list):
        proj = jnp.kron(Id4, p)
        rho_new = (proj.dot(rho_t)).dot(proj)
        prob = rho_new.trace()
        rho_new = rho_new / (rho_new.trace() + 1e-8)
        rf = partial_trace(rho_new)
        c_arr = index_update(c_arr, index[i_p], concurrence(rf))
        p_arr = index_update(p_arr, index[i_p], prob)
        rf_arr = index_update(rf_arr, index[i_p, :, :], rf)

    idx_pmax = jnp.argmax(jnp.real(c_arr))
    c_max = jnp.real(c_arr)[idx_pmax]
    p_cmax = p_arr[idx_pmax]
    rf_out = rf_arr[idx_pmax]
    # print(c_arr)
    return c_max, p_cmax, rf_out, idx_pmax


def purified_dms(rhos: np.ndarray, gate: np.ndarray):
    def sample_single_concurrence(dm):
        return purification(dm, gate)[2]

    return jax.vmap(sample_single_concurrence)(rhos)


# grid type

def double_Integral(xmin, xmax, ymin, ymax, nx, ny, A):
    dS = ((xmax - xmin) / (nx - 1)) * ((ymax - ymin) / (ny - 1))

    A_Internal = A[1:-1, 1:-1]

    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (A[0, 1:-1], A[-1, 1:-1], A[1:-1, 0], A[1:-1, -1])

    # corners
    (A_ul, A_ur, A_dl, A_dr) = (A[0, 0], A[0, -1], A[-1, 0], A[-1, -1])

    return dS * (jnp.sum(A_Internal)
                 + 0.5 * (jnp.sum(A_u) + jnp.sum(A_d) + jnp.sum(A_l) + jnp.sum(A_r))
                 + 0.25 * (A_ul + A_ur + A_dl + A_dr))

    # Rayleigh distribution, 2D, rotational symmetry


def f(x, y, sigma=0.2):
    return 1  # (jnp.sqrt(x ** 2 + y ** 2) / sigma ** 2) * jnp.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def av_inconcurrence(n: jnp.int64, a: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray):
    v1 = jnp.array([0, -1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0], dtype=jnp.complex128)
    # v2 = np.array([0, 1/math.sqrt(2), 1/math.sqrt(2), 0], dtype=np.complex)
    v3 = jnp.array([-1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)], dtype=jnp.complex128)
    # v4 = np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], dtype=np.complex)

    xy_points = jnp.array([y, x]).T

    def sample_single_concurrence(p):
        rho = (1 + p[0]) / 2 * jnp.outer(v1, v1) + (1 - p[0]) / 2 * jnp.outer(v3, v3) \
              + 1j * p[1] / 2 * jnp.outer(v3, v1) - 1j * p[1] / 2 * jnp.outer(v1, v3)
        return (1 - purification(rho)) * f(p[0], p[1])

    A = jax.vmap(sample_single_concurrence)(xy_points).reshape(n, n)
    N = jax.vmap(f)(x, y).reshape(n, n)

    norm_c = double_Integral(0, 1, 0, 2 * jnp.pi, n, n, N)
    # print(A)

    return double_Integral(0, 1, 0, 2 * jnp.pi, n, n, A) / norm_c


def generate_rhos(x: jnp.ndarray, y: jnp.ndarray):
    xy_points = jnp.array([y, x]).T

    v1 = jnp.array([0, -1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0], dtype=jnp.complex128)
    # v2 = np.array([0, 1/math.sqrt(2), 1/math.sqrt(2), 0], dtype=np.complex)
    v3 = jnp.array([-1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)], dtype=jnp.complex128)

    # v4 = np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], dtype=np.complex)

    def rho(p):
        dm = (1 + p[0]) / 2 * jnp.outer(v1, v1) + (1 - p[0]) / 2 * jnp.outer(v3, v3) \
             + 1j * p[1] / 2 * jnp.outer(v3, v1) - 1j * p[1] / 2 * jnp.outer(v1, v3)
        return dm

    return jax.vmap(rho)(xy_points)


def sampled_av_inconcurrence(a: jnp.ndarray, dm_batch: np.ndarray):
    gate = general_U(a)

    def sample_single_concurrence(dm):
        return 1 - purification(dm, gate)[0]

    A = jax.vmap(sample_single_concurrence)(dm_batch)
    # print(A)

    return jnp.mean(A, axis=0)


def sampled_av_probability(a: jnp.ndarray, dm_batch: np.ndarray):
    gate = general_U(a)

    def sample_single_concurrence(dm):
        return 1 - purification(dm, gate)[1]

    A = jax.vmap(sample_single_concurrence)(dm_batch)
    # print(A)

    return jnp.mean(A, axis=0), jnp.sqrt(jnp.var(A, axis=0))


def cnot_av_inconcurrence(x: jnp.ndarray, y: jnp.ndarray):
    dm_batch = generate_rhos(x, y)
    gate = CNOT

    def sample_single_concurrence(dm):
        return 1 - purification(dm, gate)[0]

    A = jax.vmap(sample_single_concurrence)(dm_batch)
    # print(A)

    return jnp.sum(A, axis=0) / x.shape[0]


def optimize_protocol():
    # Sample points on the circle
    n_points = 1000
    n_iter = 4
    n_guesses = 5
    n_components = 15

    x_array = np.zeros(n_points)
    y_array = np.zeros(n_points)
    f_arr = np.linspace(0.5, 1, n_points)
    t_arr = np.linspace(0.61, 1., n_points)

    dm_start = jax.vmap(werner)(f_arr)
    # dm_start = jax.vmap(rho_zsolt)(t_arr)

    for i_rand in range(n_points):
        t = 2 * np.pi * np.random.random()
        r = np.sqrt(np.random.uniform(0.0 ** 2, 1 ** 2))
        x, y = r * np.cos(t), r * np.sin(t)
        x_array[i_rand] = x
        y_array[i_rand] = y

    # x_array = np.array([0.5])
    # y_array = np.array([np.sqrt(0.5)])
    dm_start = generate_rhos(x_array, y_array)
    gate_params = [np.zeros(n_components)]
    best_fun = 1
    best_x = np.zeros(n_components)
    new_rhos = dm_start
    plot_lines = []

    for i in range(n_iter):

        def average_cost_function(a):
            # r_out = purified_dms(new_rhos, general_U(a[:n_components]))
            av_c = sampled_av_inconcurrence(a, new_rhos)
            # new_rhos = purified_dms(new_rhos, general_U(best_x))[:, 0, :, :]

            return av_c

        def average_prob(a):

            av_p = sampled_av_probability(a, new_rhos)
            return av_p

        cost_grad = grad(average_cost_function)

        # def fun_call(x):
        #    print(average_cost_function(x))

        print(best_x, best_fun)
        print(f"Initial concurrence: {1 - average_cost_function(gate_params[0])}")
        print(average_prob(gate_params[0]))
        # if i == 0:
        #    plot_lines.append(1 - average_cost_function(gate_params[0]))
        # best_fun = average_cost_function(gate_params[0])

        for i in range(n_guesses):
            x0 = 2*np.pi*np.random.randn(n_components)
            res = minimize(average_cost_function, x0=x0, method="l-bfgs-b", jac=cost_grad)
            print(res.fun)
            print(res.x)
            print(average_prob(res.x))
            if res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x

        plot_lines.append(best_fun)
        print(expm(1j * general_U(best_x)) / 0.35)
        # gate_params.append(best_x)
        new_rhos = purified_dms(new_rhos, general_U(best_x))

    plt.style.use("seaborn-colorblind")
    plt.rc('lines', linewidth=3)
    plt.title("Average concurrence plot")
    from cycler import cycler
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycler = cycler('color', colors)
    y = []
    for i_p, p in enumerate(plot_lines):
        y.append(1 - p)
        plt.axhline(y=1 - p, label=f"Concurrence after protocol iteration {i_p + 2}",
                    linestyle='-', color=colors[i_p])

    plt.rc('axes', prop_cycle=color_cycler)
    # plt.axhline(y=plot_lines[0], label="Initial average concurrence", linestyle='-', color='r')
    plt.clf()
    plt.plot(np.arange(0, n_iter), y, label="Concurrence for each protocol iteration")
    plt.xlabel("Iterations with random restart")
    plt.ylabel(r"$C$")
    plt.legend()
    plt.show()
    plt.savefig("average_concurrence.png")


def main():
    # partitions for the integral
    n = 100
    # scale parameter of the Rayleigh distribution <1, to have a peak before xy edge states x^2+y^2=1
    sigma = 0.2

    # Sampling on the sphere
    x_array = np.zeros(n ** 2)
    y_array = np.zeros(n ** 2)

    t = np.linspace(0, 2 * np.pi, n)
    r = np.linspace(0.0, 0.99, n)

    """for i_r in range(n):
        for j_t in range(n):
            x, y = r[i_r] * np.cos(t[j_t]), r[i_r] * np.sin(t[j_t])
            x_array[n * i_r + j_t] = x
            y_array[n * i_r + j_t] = y"""

    for i_rand in range(n ** 2):
        t = 2 * np.pi * np.random.random()
        r = np.sqrt(np.random.uniform(0.0 ** 2, 1 ** 2))
        x, y = r * np.cos(t), r * np.sin(t)
        x_array[i_rand] = x
        y_array[i_rand] = y

    # plt.scatter(x_array, y_array, label="Sampled points")
    # plt.show()

    def average_cost_function(a):
        return sampled_av_inconcurrence(a, x_array, y_array)

    def average_cnot_cf(a):
        return cnot_av_inconcurrence(x_array, y_array)

    cost_grad = grad(average_cost_function)

    plotlist = []

    def fun_call(x):
        plotlist.append(1 - average_cost_function(x))
        print(average_cost_function(x))

    x0 = np.zeros(8)
    print(f"Initial inconcurrence: {1 - average_cost_function(x0)}")
    print(f"CNOT inconcurrence: {1 - average_cnot_cf(x0)}")

    for i in range(10):
        x0 = np.random.randn(15)
        res = minimize(average_cost_function, x0=x0, method="l-bfgs-b", callback=fun_call, jac=cost_grad)

    plt.style.use("seaborn-colorblind")
    plt.rc('lines', linewidth=3)
    plt.title("Average concurrence plot")
    plt.plot(plotlist, label="Average concurrence")
    plt.axhline(y=1 - average_cost_function(np.zeros(15)), label="Initial concurrence", linestyle='-', color='b')
    plt.axhline(y=1 - average_cnot_cf(x0), label="CNOT concurrence", linestyle='-', color='r')
    plt.xlabel("Iterations with random restart")
    plt.ylabel(r"$C$")
    plt.legend()
    plt.ion()
    plt.savefig("average_concurrence.png")


if __name__ == "__main__":
    optimize_protocol()
