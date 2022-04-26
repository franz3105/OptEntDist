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
import jax.scipy as jsp
import os
import json

from jax import lax
from cycler import cycler
from scipy.optimize import minimize
from jax import grad
from functools import partial
from qutip_wrapper import werner, rho_zsolt, transformed_werner, tranformed_mau_state
from jax_minimize_wrapper import minimize
from su4gate import gate

# from su4gate import gate

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)
jax.config.update("jax_debug_nans", True)

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

g2 = jnp.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.complex128)
g3 = jnp.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.complex128)
g5 = jnp.array([[0, 0, -1j, 0], [0, 0, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.complex128)
g8 = (1 / jnp.sqrt(3)) * jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, -2, 0, 0], [0, 0, 0, 0]], dtype=jnp.complex128)
g10 = jnp.array([[0, 0, 0, -1j], [0, 0, 0, 0], [0, 0, 0, 0], [1j, 0, 0, 0]], dtype=jnp.complex128)
g15 = (1 / jnp.sqrt(6)) * jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -3]], dtype=jnp.complex128)

gell_man_generators = (g2, g3, g5, g8, g10, g15)
gen_seq = [g3, g2, g3, g5, g3, g10, g3, g2, g3, g5, g3, g2, g3, g8, g15]
hp_rect = np.pi * np.array(
    [1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / np.sqrt(3), 1 / np.sqrt(6), 2 * np.pi])
CNOT = jnp.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=jnp.complex128)


def get_generators():
    clifford_set = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    generators = []
    for sigma_1 in clifford_set:
        for sigma_2 in clifford_set:
            generators.append(jnp.array(qt.tensor(sigma_2, sigma_1).full()))

    del (generators[0])

    return generators


generators = get_generators()


@jax.custom_vjp
def safe_eigh(x):
    return jnp.linalg.eigh(x)


def safe_eigh_fwd(x):
    w, v = safe_eigh(x)
    return (w, v), (w, v)


def safe_eigh_bwd(res, g):
    w, v = res
    wct, vct = g
    deltas = w[..., jnp.newaxis, :] - w[..., :, jnp.newaxis]
    on_diagonal = jnp.eye(w.shape[-1], dtype=bool)
    F = jnp.where(on_diagonal, 0, 1 / jnp.where(on_diagonal, 1, deltas))
    matmul = partial(jnp.matmul, precision=lax.Precision.HIGHEST)
    vT_ct = matmul(v.T.conj(), vct)
    F_vT_vct = jnp.where(vT_ct != 0, F * vT_ct, 0)  # ignore values that would give NaN
    g = matmul(v, matmul(jnp.diag(wct) + F_vT_vct, v.T.conj()))
    g = (g + g.T.conj()) / 2
    return g,


def concurrence(rho: np.ndarray):
    YY = jnp.kron(Y, Y)
    rho_tilde = (rho.dot(YY)).dot(rho.conj().dot(YY))
    # print(rho)
    eigs = jnp.abs(jnp.sort(jnp.real(-jnp.linalg.eigvals(rho_tilde))))
    # print(eigs)
    # eigs = np.clip(eigs, 0, 1)
    comp_arr = jnp.array([0., jnp.sqrt(eigs[0]) - jnp.sqrt(eigs[1]) - jnp.sqrt(eigs[2]) - jnp.sqrt(eigs[3])])

    return jnp.max(comp_arr)


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


def gate_2(a: np.ndarray):
    # print(s)
    o = tensor([jnp.zeros((2, 2), dtype=jnp.complex128)] * 2)
    for i in range(a.shape[0]):
        o += a[i] * generators[i]

    return jsp.linalg.expm(1j * o)


def gell_man_U(a: np.ndarray):
    o = tensor([jnp.zeros((2, 2), dtype=jnp.complex128)] * 2)

    for i in range(a.shape[0]):
        o = jsp.linalg.expm(1j * a[i] * gen_seq[i]).dot(o)

    return o


def gen_SU4(a: np.ndarray):
    return gate(*list(a[:15])) * jnp.exp(1j * a[15])


def apply_entangling_gate(rho: np.ndarray, gate: np.ndarray):
    # if a is not 3D vector, then this has to be changed
    G = gate
    Gc = gate.conjugate().T

    # print(G)
    rho_f = jnp.kron(rho, rho)
    U = jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=jnp.complex128)
    UU = jnp.kron(jnp.kron(Id2, U), Id2)

    # side A
    MA = jnp.kron(G, Id4)
    MAc = jnp.kron(Gc, Id4)
    UA = (UU.dot(MA)).dot(UU)
    UAc = (UU.dot(MAc)).dot(UU)

    # side B
    MB = jnp.kron(Id4, G)
    MBc = jnp.kron(Id4, Gc)
    UB = (UU.dot(MB)).dot(UU)
    UBc = (UU.dot(MBc)).dot(UU)

    rho_t = (((UA.dot(UB)).dot(rho_f)).dot(UBc)).dot(UAc)

    return rho_t


def purification(rho: np.ndarray, gate: np.ndarray):
    rho_t = apply_entangling_gate(rho, gate)
    # print(gate)

    proj_list = [p00, p01]
    c_arr = jnp.array([0, ] * len(proj_list), dtype=jnp.float64)
    p_arr = jnp.array([0, ] * len(proj_list), dtype=jnp.float64)
    rf_arr = jnp.zeros((len(proj_list), rho.shape[0], rho.shape[1]), dtype=jnp.complex128)

    for i_p, p in enumerate(proj_list):
        proj = jnp.kron(Id4, p)
        rho_new = (proj.dot(rho_t)).dot(proj)
        prob = jnp.real((rho_new.trace() + 1e-6))
        # print(rho_new)
        # print(rho_new.trace())
        rho_new = rho_new / (rho_new.trace() + 1e-6)
        rf = partial_trace(rho_new)
        # print(concurrence(rf))
        c_arr = index_update(c_arr, index[i_p], concurrence(rf))
        p_arr = index_update(p_arr, index[i_p], prob)
        rf_arr = index_update(rf_arr, index[i_p, :, :], rf)

    idx_pmax = jnp.argmax(c_arr)
    # idx_pmin = jnp.argmin(c_arr)
    c_max = jnp.real(c_arr)[idx_pmax]
    # print(c_max)
    # print(jnp.round(jnp.real(p_arr),3))
    p_cmax = p_arr[idx_pmax]
    rf_out = rf_arr[idx_pmax]

    # print(c_max)

    return c_max, jnp.real(p_cmax), rf_out, idx_pmax


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


def av_inconcurrence(n: jnp.int64, x: jnp.ndarray, y: jnp.ndarray):
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
        dm = jnp.array([[(1 - p[0]) / 4, (1j * p[1]) / 4, -(1j * p[1]) / 4, 1 / 4 * (-1 + p[0])],
                        [-(1j * p[1]) / 4, (1 + p[0]) / 4, 1 / 4 * (-1 - p[0]), (1j * p[1]) / 4],

                        [(1j * p[1]) / 4, 1 / 4 * (-1 - p[0]), (1 + p[0]) / 4, -(1j * p[1]) / 4],
                        [1 / 4 * (-1 + p[0]), -(1j * p[1]) / 4, (1j * p[1]) / 4, (1 - p[0]) / 4]], jnp.complex128)

        return dm

    return jax.vmap(rho)(xy_points)


def f(x):
    return 1


def mauricio_av_inconcurrence(a: jnp.ndarray, x_arr: np.ndarray):
    gate_a = gen_SU4(a)

    def sample_single_concurrence(x):
        dm = tranformed_mau_state(x)
        return (1 - purification(dm, gate_a)[0]) * f(x)

    A = jax.vmap(sample_single_concurrence)(x_arr)

    return jnp.mean(A, axis=0)


def sampled_av_inconcurrence(a: jnp.ndarray, dm_batch: np.ndarray, pdf=None):
    gate = gen_SU4(a)

    def sample_single_concurrence(dm):
        return 1 - purification(dm, gate)[0]

    A = jax.vmap(sample_single_concurrence)(dm_batch)
    # print(A)

    return jnp.mean(A, axis=0)


def sampled_av_probability(a: jnp.ndarray, dm_batch: np.ndarray):
    gate = gen_SU4(a)

    # print(gate.dot(gate.T.conjugate()))

    def sample_single_concurrence(dm):
        return 2 * purification(dm, gate)[1]

    A = jax.vmap(sample_single_concurrence)(dm_batch)
    # print(A)

    return A, jnp.sqrt(jnp.var(A, axis=0))


def cnot_av_inconcurrence(x: jnp.ndarray, y: jnp.ndarray):
    dm_batch = generate_rhos(x, y)
    gate = CNOT

    def sample_single_concurrence(dm):
        return 1 - purification(dm, gate)[0]

    A = jax.vmap(sample_single_concurrence)(dm_batch)
    # print(A)

    return jnp.sum(A, axis=0) / x.shape[0]


def werner_sample(werner_state, n_points, f_min, f_max):
    f_arr = np.linspace(f_min, f_max, n_points)

    dm_sample = jax.vmap(werner_state)(f_arr)

    return dm_sample, (f_arr,)


def mau_state_sample(mau_state, n_points, c_min, c_max):
    c_arr = np.linspace(c_min, c_max, n_points)

    dm_sample = jax.vmap(mau_state)(c_arr)

    return dm_sample, (c_arr,)


def xy_state_sample(n_points, x_min, x_max):
    x_array = np.zeros(n_points)
    y_array = np.zeros(n_points)

    for i_rand in range(n_points):
        t = 2 * np.pi * np.random.random()
        r = np.sqrt(np.random.uniform(x_min ** 2, x_max ** 2))
        x, y = r * np.cos(t), r * np.sin(t)
        x_array[i_rand] = x
        y_array[i_rand] = y

    dm_sample = generate_rhos(x_array, y_array)

    return dm_sample, (x_array, y_array)


def xy_state(x_array, y_array):
    dm_sample = generate_rhos(x_array, y_array)
    return dm_sample, (x_array, y_array)


def xy_state_rot(x_array, y_array):
    v1 = jnp.array([0, -1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0], dtype=jnp.complex128)
    # v2 = np.array([0, 1/math.sqrt(2), 1/math.sqrt(2), 0], dtype=np.complex)
    v3 = jnp.array([-1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)], dtype=jnp.complex128)

    def rho(p):
        dm = (1 + jnp.sqrt(p[0] ** 2 + p[1] ** 2)) / 2 * jnp.outer(v1, v1) + \
             (1 - jnp.sqrt(p[0] ** 2 + p[1] ** 2)) / 2 * jnp.outer(v3, v3)
        return dm

    dms = jax.vmap(rho)(jnp.array([y_array, x_array]).T)
    return dms, (x_array, y_array)


def optimize_protocol(dm_start, points, n_iter, n_guesses, n_components):
    # Sample points on the circle

    gate_params = [np.zeros(n_components + 1, dtype=jnp.float64)]
    n_points = dm_start.shape[0]
    best_fun = 1
    best_x = np.zeros(n_components)
    best_ing = None
    new_rhos = dm_start
    opt_curves = np.zeros((5000, n_iter, n_guesses), dtype=jnp.float64)
    init_concurrence = []
    plot_lines = []
    all_probs = np.zeros((n_points, n_iter))
    best_concurrences = np.zeros(n_iter)
    best_unitaries = np.zeros((n_iter, 16), dtype=np.complex128)
    prob_avg = []
    np.random.seed(0)

    for it in range(n_iter):
        print(f"Iteration {it + 1}/{n_iter}")

        def average_cost_function(a):
            # r_out = purified_dms(new_rhos, general_U(a[:n_components]))
            av_c = sampled_av_inconcurrence(a, new_rhos)
            # new_rhos = purified_dms(new_rhos, general_U(best_x))[:, 0, :, :]

            return av_c

        def all_prob(a):

            av_p = sampled_av_probability(a, new_rhos)[0]
            return av_p

        cost_grad = grad(average_cost_function)

        c_start = 1 - average_cost_function(gate_params[0])
        print(c_start)
        init_concurrence.append(c_start)

        # if i == 0:
        #    plot_lines.append(1 - average_cost_function(gate_params[0]))
        # best_fun = average_cost_function(gate_params[0])
        best_fun = 1 - c_start

        for i_ng in range(n_guesses):

            def save_plot(v):
                idx = np.count_nonzero(opt_curves[:, it, i_ng])
                opt_curves[idx, it, i_ng] = (1 - average_cost_function(v))

            x0 = np.random.uniform(np.zeros(n_components + 1), hp_rect, size=(n_components + 1,))
            res = minimize(average_cost_function, x0=x0, method="l-bfgs-b", jac=cost_grad, callback=save_plot)
            print(res.fun)
            print(res.x)
            if res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x
                best_ing = i_ng

            # print(best_fun)
            if best_fun < 1e-2:
                break

        non_zero_entries = np.count_nonzero(opt_curves[:, it, best_ing])
        all_probs[:, it] = all_prob(best_x)
        plot_lines.append(opt_curves[:non_zero_entries, it, best_ing])
        print(f"Concurrence: {1 - average_cost_function(best_x)}")
        gate_params.append(best_x)
        new_rhos = purified_dms(new_rhos, gen_SU4(best_x))
        best_unitaries[it, :] = gen_SU4(best_x).flatten()
        print(gen_SU4(best_x))
        best_concurrences[it] = 1 - best_fun
        prob_avg.append(np.mean(all_prob(best_x), axis=0))

    return np.array(plot_lines, dtype=object), np.array(best_concurrences), np.array(np.real(prob_avg)), np.real(
        all_probs), \
           best_unitaries, points


def main():
    cwd = os.getcwd()
    data_folder = os.path.join(cwd, "data")
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    else:
        pass

    # List of names of avaialble states
    # state_names = ["Werner"]
    state_names = ["XY_single"]
    n_points = 1
    n_iter = 5
    n_guesses = 5
    n_components = 15

    for sn in state_names:
        if sn == "Werner":
            dm_start, points = werner_sample(werner, n_points, 0.5, 1.0)
        elif sn == "Transformed Werner":
            dm_start, points = werner_sample(transformed_werner, n_points, 0.51, 0.99)
        elif sn == "Mauricio":
            dm_start, points = mau_state_sample(rho_zsolt, n_points, 0.51, 0.99)
        elif sn == "Transformed Mauricio":
            dm_start, points = mau_state_sample(tranformed_mau_state, n_points, 0.51, 0.99)
        elif sn == "XY_avg":
            dm_start, points = xy_state_sample(n_points, 0.1, 0.95)
        elif sn == "XY_single":
            x_array = np.array([0.25])
            y_array = np.array([0.25])
            dm_start, points = xy_state_rot(x_array, y_array)
        else:
            raise NotImplementedError("This type of state is not implemented!")
        hyperparams = dict(state_name=sn, n_points=n_points, n_iter=n_iter, n_guesses=n_guesses,
                           n_components=n_components)

        sn_folder_path = os.path.join(data_folder, f"{sn}")
        if not os.path.exists(sn_folder_path):
            os.mkdir(sn_folder_path)
        else:
            pass

        plot_lines, best_concurrences, probabilities, all_probs, best_unitaries, data_points = \
            optimize_protocol(dm_start, points, n_iter, n_guesses, n_components)

        # np.savetxt(os.path.join(sn_folder_path, "plotlines.txt"), plot_lines)
        np.savetxt(os.path.join(sn_folder_path, "best_concurrences.txt"), best_concurrences)
        np.savetxt(os.path.join(sn_folder_path, "avg_probabilities.txt"), probabilities)
        np.savetxt(os.path.join(sn_folder_path, "probabilities.txt"), all_probs)
        np.savetxt(os.path.join(sn_folder_path, "data/best_unitaries.txt"), best_unitaries)
        np.savetxt(os.path.join(sn_folder_path, "state_parameters_point.txt"), np.vstack(data_points))
        with open(os.path.join(sn_folder_path, 'opt_data.json'), 'w') as fp:
            json.dump(hyperparams, fp)

        plot(sn_folder_path)

    return


def plot(data_folder):
    # plot_lines = np.loadtxt(os.path.join(data_folder, "plotlines.txt"))
    plt.clf()

    best_concurrences = np.loadtxt(os.path.join(data_folder, "best_concurrences.txt"))
    probs = np.loadtxt(os.path.join(data_folder, "avg_probabilities.txt"))
    all_probs = np.loadtxt(os.path.join(data_folder, "probabilities.txt"))
    state_data_points = np.loadtxt(os.path.join(data_folder, "state_parameters_point.txt"))
    with open(os.path.join(data_folder, 'opt_data.json'), 'r') as fp:
        hyperparams = json.load(fp)

    plt.style.use("seaborn-colorblind")
    plt.rc('lines', linewidth=2)
    # plt.title("Average concurrence plot")
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams['figure.constrained_layout.use'] = True
    color_cycler = cycler('color', colors)

    # plot_lines = np.concatenate(plot_lines, axis=0).flatten()

    # for i_c, c in enumerate(plot_lines): print(c) plt_idx = np.count_nonzero(c) print(plt_idx_init,
    # plt_idx) plt.plot(np.arange(plt_idx_init,  plt_idx_init + plt_idx), c[:plt_idx], label=f"Concurrence after
    # protocol iteration {i_c + 2}", linestyle='-', color=colors[i_c]) plt_idx_init = plt_idx + plt_idx_init
    # plt.figure(0)
    # plt.plot(plot_lines, linestyle='-')
    # plt.rc('axes', prop_cycle=color_cycler)
    # plt.axhline(y=init_concurrence[0], label="Initial average concurrence", linestyle='--', alpha=0.5)
    # plt.axhline(y=plot_lines[-1], label=f"Maximal concurrence after {n_iter} iterations", linestyle='--', alpha=0.5)
    # plt.xlabel("Iterations with random restart")
    # plt.ylabel(r"$C$")
    # plt.legend()
    # plt.ion()
    # plt.savefig("optim_plot.png")
    x_data = np.arange(1, best_concurrences.shape[0] + 1)

    plt.figure(1)
    plt.plot(x_data, best_concurrences, "o-", linewidth=2, label="Average concurrence")
    plt.xlabel("Number of protocol steps")
    plt.ylabel(r"$C$")
    plt.xticks(range(1, best_concurrences.shape[0] + 1))
    plt.legend()
    # plt.ion()
    plt.savefig(os.path.join(data_folder, "data/average_concurrence.png"))

    plt.figure(2)
    plt.plot(x_data, probs, "o-", linewidth=2, label="Average probability")
    plt.xlabel("Number of protocol steps")
    plt.ylabel(r"$p$")
    plt.xticks(range(1, best_concurrences.shape[0] + 1))
    plt.legend()
    # plt.ion()
    plt.savefig(os.path.join(data_folder, "data/average_probability.png"))

    if hyperparams["state_name"] != "XY":
        # 2D plot for other states
        plt.figure(3)
        for it in range(hyperparams["n_iter"]):
            plt.plot(state_data_points, all_probs[:, it], "o-", linewidth=2,
                     label=f"Iteration {it + 1}")
        plt.xlabel("F")
        plt.ylabel(r"$p$")
        plt.legend()
        # plt.ion()
        plt.savefig(os.path.join(data_folder, "probability.png"))
    else:
        # 3D plot for XY state
        fig = plt.figure(3)
        ax = fig.gca(projection='3d')
        for it in range(hyperparams["n_iter"]):
            plt.plot(state_data_points, all_probs[:, it], "o-", linewidth=2,
                     label=f"Iteration {it + 1}")

            ax.plot_trisurf(state_data_points[0, :], state_data_points[1, :], all_probs[:, it], linewidth=0.2,
                            antialiased=True)

        ax.xlabel("x")
        ax.ylabel("y")
        ax.zlabel(r"$p$")
        plt.legend()
        # plt.ion()
        plt.savefig(os.path.join(data_folder, "probability.png"))

    plt.close()


def test_unitary():
    u = np.loadtxt('best_unitaries.txt', dtype=np.complex_).reshape(8, 4, 4)
    for i in range(5):
        print("Unitary:", u[i, ::])
        print("Identity:", np.round(u[i, ::].dot(u[i, ::].T.conjugate()), 2))


if __name__ == "__main__":
    main()
