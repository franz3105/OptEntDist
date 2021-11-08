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
from scipy.integrate import simps
from scipy.optimize import minimize
from jax.ops import index, index_update
from jax import jit, partial

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


@jit
def concurrence(rho: np.ndarray):
    YY = jnp.kron(Y, Y)
    rho_tilde = (YY.dot(rho.T)).dot(YY)
    eigs = -jnp.sort(-jnp.linalg.eigvals(rho.dot(rho_tilde)))
    comp_arr = jnp.array([0., jnp.sqrt(eigs[0]) - jnp.sum(jnp.sqrt(eigs[1:]))])

    return jnp.max(jnp.real(comp_arr))


@jit
def partial_trace(rho: np.ndarray, subscripts='ijklmnkl->ijmn'):
    rho_rs = jnp.reshape(rho, jnp.array([2, 2, 2, 2, 2, 2, 2, 2]))
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


@jit
def general_U(s: np.ndarray):
    o = tensor([jnp.zeros((2, 2))] * 2)
    for i in range(s.shape[0]):
        o += s[i] * generators[i]

    return o


@jit
def purification(rho: np.ndarray, a: np.ndarray):
    # if a is not 3D vector, then this has to be changed
    G = general_U(a)

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

    c = jnp.array([0, 0, 0, 0], dtype=jnp.complex128)

    # 11 result on A2B2
    proj_list = [p00, p01, p10, p11]

    for i_p, p in enumerate(proj_list):
        proj = jnp.kron(Id4, p)
        rho_new = (proj.dot(rho_t)).dot(proj)
        rho_new = rho_new / (rho_new.trace() + 1e-8)
        rf = partial_trace(rho_new)
        c = index_update(c, index[i_p], concurrence(rf))

    return jnp.max(jnp.real(c))


# grid type
@jit
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


@jit
def f(x, y):
    return 1


@partial(jit, static_argnums=(0,))
def av_inconcurrence(n, a: jnp.ndarray, x, y):
    v1 = jnp.array([0, -1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0], dtype=jnp.complex128)
    # v2 = np.array([0, 1/math.sqrt(2), 1/math.sqrt(2), 0], dtype=np.complex)
    v3 = jnp.array([-1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)], dtype=jnp.complex128)
    # v4 = np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], dtype=np.complex)

    xy_points = jnp.array([y, x]).T

    @jit
    def sample_single_concurrence(p):
        rho = (1 + p[0]) / 2 * jnp.outer(v1, v1) + (1 - p[0]) / 2 * jnp.outer(v3, v3) \
              + 1j * p[1] / 2 * jnp.outer(v3, v1) - 1j * p[1] / 2 * jnp.outer(v1, v3)
        return (1 - purification(rho, a)) * f(p[0], p[1])

    A = jax.vmap(sample_single_concurrence)(xy_points).reshape(n, n)
    N = jax.vmap(f)(x, y).reshape(n, n)

    norm_c = double_Integral(0, 1, 0, 2 * jnp.pi, n, n, N)
    # print(norm_c)

    return double_Integral(0, 1, 0, 2 * jnp.pi, n, n, A) / norm_c


@partial(jit, static_argnums=(0,))
def av_est_concurrence(n, a: jnp.ndarray):
    c = 0
    for i in range(n):
        for j in range(n):
            c += f(x, y) * (1 - purification(rho, a))

    return c


def main():
    # a is now just 3 dimensional
    a = jnp.array([1.5, 1.5, 1, 2, 4, 5, 6, 7], dtype=jnp.complex128)
    # print(purification(rho, a))

    # partitions for the integral
    n = 50
    # scale parameter of the Rayleigh distribution <1, to have a peak before xy edge states x^2+y^2=1
    sigma = 0.2

    # Sampling on the sphere
    x_array = np.zeros(n ** 2)
    y_array = np.zeros(n ** 2)

    for i_rand in range(n ** 2):
        t = 2 * np.pi * np.random.random()
        u = np.random.random() + np.random.random()
        r = 2 - u if u > 1 else u
        x, y = r * np.cos(t), r * np.sin(t)
        x_array[i_rand] = x
        y_array[i_rand] = y

    # print((x ** 2 + y ** 2).all() == (jnp.outer(r, r).flatten()).all())
    print(np.mean(x_array ** 2 + y_array ** 2))
    plt.scatter(x_array, y_array, label="Sampled points")
    plt.show()

    @jit
    def average_cost_function(a):
        return av_inconcurrence(n, a, x_array, y_array)

    plotlist = []

    def fun_call(x):
        plotlist.append(average_cost_function(x))
        print(average_cost_function(x))

    x0 = np.zeros(8)
    print(f"Initial inconcurrence: {average_cost_function(x0)}")

    for i in range(10):
        x0 = np.random.randn(15)
        res = minimize(average_cost_function, x0=x0, method="l-bfgs-b", callback=fun_call)

    print(res.fun)
    plt.plot(plotlist, label="Average inconcurrence")
    plt.xlabel("Iterations with random restart")
    plt.plot("Average inconcurrence")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
