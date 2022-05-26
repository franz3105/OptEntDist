#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 08:32:24 2021

@author: zsolt
"""
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import qutip as qt
import jax.scipy as jsp
import os
import json

from jax import value_and_grad, jit
from cycler import cycler
from qutip_wrapper import tranformed_mau_state
from jax_minimize_wrapper import minimize_jax
from su4gate import gate

# JAX configuration
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)
jax.config.update("jax_debug_nans", True)

# ----------------------------------------------------------------------------------------------------------------------

# General definition for states, bases, projectors and gates

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
    [1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1, 1 / np.sqrt(3), 1 / np.sqrt(6)])
CNOT = np.exp(1j*np.pi/2)*jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex128)


# ----------------------------------------------------------------------------------------------------------------------

def get_generators():
    """ Creates standared generators for SU(4) """
    clifford_set = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    generators = []
    for sigma_1 in clifford_set:
        for sigma_2 in clifford_set:
            generators.append(jnp.array(qt.tensor(sigma_2, sigma_1).full()))

    del (generators[0])

    return generators

#@jit
def concurrence(rho: np.ndarray):
    """
    Function computing the concurrence
    :param rho: Input density matrix
    :return: concurrence

    """
    YY = jnp.kron(Y, Y)
    rho_tilde = (rho.dot(YY)).dot(rho.conj().dot(YY))
    eigs = -jnp.linalg.eigvals(rho_tilde)
    eigs = jnp.abs(jnp.sort(jnp.real(eigs))) + 1e-8

    #eigs = jnp.clip(eigs, 1e-6, 1)  # Bounds for numerical stability
    comp_arr = jnp.array([0., jnp.sqrt(eigs[0]) - jnp.sqrt(eigs[1]) - jnp.sqrt(eigs[2]) - jnp.sqrt(eigs[3])])
    #print(comp_arr)

    return jnp.max(comp_arr)

#@jit
def partial_trace(rho: np.ndarray, subscripts='ijklmnkl->ijmn'):
    """

    :param rho: Input density matrix
    :param subscripts: superscripts describing the partial trace
    :return: partial trace
    """
    rho_rs = jnp.reshape(rho, [2, 2, 2, 2, 2, 2, 2, 2])
    rho_pt = jnp.einsum(subscripts, rho_rs).reshape(4, 4)

    return rho_pt


def tensor(qubit_list):
    """

    :param qubit_list: List of qubits [q_1, .., q_N]
    :return: tensor product of all qubits
    """
    s = None

    for i_el, el in enumerate(qubit_list):
        if i_el == 0:
            s = el
        else:
            s = jnp.kron(s, el)

    return s

#@jit
def gell_man_U(a: np.ndarray):
    """
    General SU(4) Euler parametrization (see su4gate.py)
    :param a: gate parameter array.
    :return: general SU(4) unitary
    """
    o = tensor([jnp.zeros((2, 2), dtype=jnp.complex128)] * 2)

    for i in range(a.shape[0]):
        o = jsp.linalg.expm(1j * a[i] * gen_seq[i]).dot(o)

    return o

#@jit
def gen_SU4(a: np.ndarray):
    """
    Analytic expression of a SU(4) gate derived from Mathematica.
    :param a: gate parameter array.
    :return: general SU(4) unitary
    """

    return gate(*list(a))

#@jit
def apply_entangling_gate(rho: np.ndarray, unitary_gate: np.ndarray):
    """

    :param rho: Input density matrix on system A (or B).
    :param unitary_gate: Entangling gate.
    :return: New density matrix after operation.
    """
    # if a is not 3D vector, then this has to be changed
    G_A = unitary_gate
    Gc_A = unitary_gate.conjugate().T

    # print(G)
    rho_f = jnp.kron(rho, rho)
    U = jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=jnp.complex128)
    UU = jnp.kron(jnp.kron(Id2, U), Id2)

    # side A
    MA = jnp.kron(G_A, Id4)
    MAc = jnp.kron(Gc_A, Id4)
    UA = (UU.dot(MA)).dot(UU)
    UAc = (UU.dot(MAc)).dot(UU)

    # side B
    MB = jnp.kron(Id4, G_A)
    MBc = jnp.kron(Id4, Gc_A)
    UB = (UU.dot(MB)).dot(UU)
    UBc = (UU.dot(MBc)).dot(UU)

    rho_t = (((UA.dot(UB)).dot(rho_f)).dot(UBc)).dot(UAc)

    return rho_t

#@jit
def purification(rho: np.ndarray, unitary_gate: np.ndarray):
    """

    :param rho: Input density matrix on system A (or B).
    :param unitary_gate: Entangling gate.
    :return: max concurrence, max probability, purified density matrix, argmax index of concurrence.
    Here we "use" an "argmax policy" for the probability.
    """
    rho_t = apply_entangling_gate(rho, unitary_gate)
    # print(unitary_gate)

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
        c_arr = c_arr.at[i_p].set(concurrence(rf))
        p_arr = p_arr.at[i_p].set(prob)
        rf_arr = rf_arr.at[i_p, :, :].set(rf)

    idx_pmax = jnp.argmax(c_arr)
    # idx_pmin = jnp.argmin(c_arr)
    c_max = jnp.real(c_arr)[idx_pmax]
    # print(c_max)
    # print(jnp.round(jnp.real(p_arr),3))
    p_cmax = p_arr[idx_pmax]
    rf_out = rf_arr[idx_pmax]

    # print(c_max)

    return c_max, jnp.real(p_cmax), rf_out, idx_pmax


def purified_dms(rhos: np.ndarray, unitary_gate: np.ndarray):
    """

    :param rhos: Batch of input density matrices.
    :param unitary_gate: Unitary entangling gate.
    :return: batch of concurrences.
    """
    def sample_single_concurrence(dm):
        return purification(dm, unitary_gate)[2]

    return jax.vmap(sample_single_concurrence)(rhos)


# grid type

def double_Integral(xmin, xmax, ymin, ymax, nx, ny, A):
    """

    :param xmin: Inferior bound x
    :param xmax: Superior bound x
    :param ymin: Inferior bound y
    :param ymax: Superior bound y
    :param nx: Number of slices in x
    :param ny: Number of slices in y
    :param A: function values
    :return: integral over area.
    """
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
    """
    Computes the average inconcurrence by evalutating the integral numerically (slow).
    :param n: Number of samples
    :param x: array of x-coordinates
    :param y: array of y-coordinates
    :return: Average concurrence computed with integral.
    """
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
    """
    Creates a batch of xy-dependent density matrices.

    :param x: array of x-coordinates
    :param y: array of y-coordinates
    :return: batch of xy-dependent density matrices.
    """
    xy_points = jnp.array([y, x]).T

    # v4 = np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], dtype=np.complex)

    def rho(p):
        dm = jnp.array([[(1 - p[0]) / 4, (1j * p[1]) / 4, -(1j * p[1]) / 4, 1 / 4 * (-1 + p[0])],
                        [-(1j * p[1]) / 4, (1 + p[0]) / 4, 1 / 4 * (-1 - p[0]), (1j * p[1]) / 4],

                        [(1j * p[1]) / 4, 1 / 4 * (-1 - p[0]), (1 + p[0]) / 4, -(1j * p[1]) / 4],
                        [1 / 4 * (-1 + p[0]), -(1j * p[1]) / 4, (1j * p[1]) / 4, (1 - p[0]) / 4]], jnp.complex128)

        return dm

    return jax.vmap(rho)(xy_points)


def f(x, sigma=1):
    return 1#1/(2*np.pi*sigma)**0.5*np.exp(0.5 * (x[0]**2 + x[1]**2) / sigma ** 2)


def mauricio_av_inconcurrence(a: jnp.ndarray, x_arr: np.ndarray):
    """
    Average inconcurrence of the transformed state that can be purified in 1 step.
    :param a: entangling gate parameters
    :param x_arr: array of state parameters.
    :return: average concurrence (sampled)
    """
    gate_a = gen_SU4(a)

    #@jit
    def sample_single_concurrence(x):
        dm = tranformed_mau_state(x)
        return (1 - purification(dm, gate_a)[0]) * f(x)

    A = jax.vmap(sample_single_concurrence)(x_arr)

    return jnp.mean(A, axis=0)


def sampled_av_inconcurrence(a: jnp.ndarray, dm_batch: np.ndarray):
    """
    Average inconcurrence of a batch of density matrices.
    :param dm_batch:  batch of density matrices
    :param a: entangling gate parameters
    :return: average concurrence (sampled)
    """
    gate_1 = gen_SU4(a)

    #@jit
    def sample_single_concurrence(dm):
        return 1 - purification(dm, gate_1)[0]

    A = jax.vmap(sample_single_concurrence)(dm_batch)

    return jnp.mean(A, axis=0)

def sampled_av_cnot_inconcurrence(dm_batch: np.ndarray):
    """
    Average inconcurrence of a batch of density matrices.
    :param dm_batch:  batch of density matrices
    :param a: entangling gate parameters
    :return: average concurrence (sampled)
    """
    gate_1 = CNOT

    #@jit
    def sample_single_concurrence(dm):
        return 1 - purification(dm, gate_1)[0]

    A = jax.vmap(sample_single_concurrence)(dm_batch)

    return jnp.mean(A, axis=0)

def sampled_av_probability(a: jnp.ndarray, dm_batch: np.ndarray):
    """
    Average probability of a batch of density matrices.
    :param dm_batch:  batch of density matrices
    :param a: entangling gate parameters
    :return: average probability and its std (sampled)
    """
    gate = gen_SU4(a)

    # print(unitary_gate.dot(unitary_gate.T.conjugate()))

    #@jit
    def sample_single_concurrence(dm):
        return 2 * purification(dm, gate)[1]

    A = jax.vmap(sample_single_concurrence)(dm_batch)

    return A, jnp.sqrt(jnp.var(A, axis=0))


def werner_sample(werner_state, n_points, f_min, f_max):
    """
    Samples werner states over a given 1d interval.
    :param werner_state: function returning Werner states as a function of their parameter (usually F)
    :param n_points: Number of samples
    :param f_min: Inferior bound for the parameter
    :param f_max: Superior bound for the parameter
    :return: batch of density matrices and tuple with the array of parameters.
    """
    f_arr = np.linspace(f_min, f_max, n_points)

    dm_sample = jax.vmap(werner_state)(f_arr)

    return dm_sample, (f_arr,)


def mau_state_sample(mau_state, n_points, c_min, c_max):
    """
    Samples mau states over a given 1d interval.
    :param mau_state: function returning mau states as a function of their parameter (x in the paper)
    :param n_points: Number of samples
    :param c_min: Inferior bound for the parameter
    :param c_max: Superior bound for the parameter
    :return: batch of density matrices and tuple with the array of parameters.
    """
    c_arr = np.linspace(c_min, c_max, n_points)

    dm_sample = jax.vmap(mau_state)(c_arr)

    return dm_sample, (c_arr,)


def xy_state_sample(n_points, r_min, r_max):
    """
     Returns batch of XY states sampled radially inside the unit circle.
    :param r_max: Minimum value of the radius
    :param r_min: Maximum value of the radius
    :param n_points: Number of samples
    :return: batch of density matrices and tuple with the arrays of parameters.
    """

    x_array = np.zeros(n_points)
    y_array = np.zeros(n_points)

    for i_rand in range(n_points):
        t = 2 * np.pi * np.random.random()
        r = np.sqrt(np.random.uniform(r_min ** 2, r_max ** 2))
        x, y = r * np.cos(t), r * np.sin(t)
        x_array[i_rand] = x
        y_array[i_rand] = y

    dm_sample = generate_rhos(x_array, y_array)

    return dm_sample, (x_array, y_array)


def xy_state(x_array, y_array):
    """
    Returns batch of XY states over a given 2d interval defined by x_array and y_array.
    :param x_array: Samples x values
    :param y_array: Samples y values
    :return: batch of density matrices, Tuple[x array, y array]
    """
    dm_sample = generate_rhos(x_array, y_array)
    return dm_sample, (x_array, y_array)


def xy_state_rot(x_array, y_array):
    """
    Returns batch of rotated XY states over a given 2d interval defined by x_array and y_array.
    :param x_array: Samples x values
    :param y_array: Samples y values
    :return: batch of density matrices, Tuple[x array, y array]
    """
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
    """

    :param dm_start: Initial batch of density matrices.
    :param points: Number of sampled points
    :param n_iter: Number of protocol iterations
    :param n_guesses: Number of guesses per iteration
    :param n_components: Number of unitary gate parameters (e.g. 15 for SU(4))
    :return: Tuple containing:
            - best_concurrences: array of the highest concurrences for each iteration.
            - prob_avg: Average probabilities.
            - all_probabilities: Probability curves for each iteration.
            - best_unitaries: Best unitaries for each iteration (arg of best_concurrences).
            - points: Tuples of arrays of parameters
    """
    # Sample points on the circle

    gate_params = [np.zeros(n_components, dtype=jnp.float64)]
    n_points = dm_start.shape[0]
    best_x = np.zeros(n_components)
    best_ing = None
    new_rhos = dm_start
    opt_curves = np.zeros((5000, n_iter+1, n_guesses), dtype=jnp.float64)
    concurrences = []
    plot_lines = []
    all_probs = np.zeros((n_points, n_iter+1))
    best_concurrences = np.zeros(n_iter+1)
    best_unitaries = np.zeros((n_iter+1, 16), dtype=np.complex128)
    prob_avg = []
    #np.random.seed(0)
    cnot = False

    for it in range(1, n_iter+1):
        print(f"Iteration {it}/{n_iter}")

        def average_cost_function(a):
            # r_out = purified_dms(new_rhos, general_U(a[:n_components]))
            #av_c = sampled_av_inconcurrence(a, new_rhos)
            if cnot:
                av_c = sampled_av_cnot_inconcurrence(new_rhos)
            else:
                av_c = sampled_av_inconcurrence(a, new_rhos)

            # new_rhos = purified_dms(new_rhos, general_U(best_x))[:, 0, :, :]

            return av_c

        def all_prob(a):

            av_p = sampled_av_probability(a, new_rhos)[0]
            return av_p

        cost_and_grad = value_and_grad(average_cost_function)

        c_start = 1 - average_cost_function(gate_params[0])

        best_fun = 1 - c_start
        print(f"Initial concurrence iteration {it}: {best_fun}")
        best_concurrences[0] = c_start
        #print(f"CNOT concurrence: {1 - sampled_av_cnot_inconcurrence(new_rhos)}")

        for i_ng in range(n_guesses):
            if it == 1:  # Use this ansatz for the XY state, then it becomes easier to find the
            # first good guess for the optimization. This was found during our experiments. It can be found also under
            # normal circumstances, but it requires running the algorithm multiple times with lots of seeds.
                x0 = np.array([1.27223927, 1.8222995,  1.01031067, 0.7565204,  1.91471685, 1.37198244,
                      0.11381644, 1.04697845, 0.92621291, 0.46467374, 1.10616393, 0.4426136,
                      1.2732354, 0.36559595, 1.35753604])
            else:
                x0 = np.random.uniform(np.zeros(n_components), hp_rect, size=(n_components,))
            res = minimize_jax(cost_and_grad, x0=x0, method="l-bfgs-b", jac=True)
            print(f"Best inconcurrence guess {i_ng}: {res.fun}")
            print(res.x)

            if res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x
                concurrences.append(1 - res.fun)
                best_ing = i_ng

            if best_fun < 1e-2:
                break

        all_probs[:, it] = all_prob(best_x)
        print(f"Best concurrence iteration {it}: {1 - average_cost_function(best_x)}")
        gate_params.append(best_x)
        best_unitaries[it, :] = gen_SU4(best_x).flatten()
        #print(gen_SU4(best_x))
        best_concurrences[it] = 1 - best_fun
        print(best_concurrences)
        prob_avg.append(np.mean(all_prob(best_x), axis=0))

        if cnot:
            new_rhos = purified_dms(new_rhos, CNOT)
        else:
            new_rhos = purified_dms(new_rhos, gen_SU4(best_x))



    res = np.array(best_concurrences), np.array(np.real(prob_avg)), np.real(
        all_probs), best_unitaries, points

    return res


def plot(data_folder):
    """
    Plots concurrence and probability curves as a function from relevant parameters from the data saved in data_folder
    :param data_folder: Folder where the data of the optimization is saved.
    :return: None
    """
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

    x_data = np.arange(1, best_concurrences.shape[0] + 1)

    plt.figure(1)
    plt.plot(x_data, best_concurrences, "o-", linewidth=2, label="Average concurrence")
    plt.xlabel("Number of protocol steps")
    plt.ylabel(r"$C$")
    plt.xticks(range(1, best_concurrences.shape[0] + 1))
    plt.legend()
    # plt.ion()
    plt.savefig(os.path.join(data_folder, "average_concurrence.png"))

    plt.figure(2)
    plt.plot(x_data, probs, "o-", linewidth=2, label="Average probability")
    plt.xlabel("Number of protocol steps")
    plt.ylabel(r"$p$")
    plt.xticks(range(1, best_concurrences.shape[0] + 1))
    plt.legend()
    # plt.ion()
    plt.savefig(os.path.join(data_folder, "average_probability.png"))

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
