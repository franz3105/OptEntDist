# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import qutip as qt
import jax.scipy as jsp
import jax.numpy as jnp
import jax
from qutip.measurement import measure, measurement_statistics
from qutip.qip.operations import controlled_gate, cnot, molmer_sorensen
import numpy as np
import matplotlib.pyplot as plt
import random
from jax.ops import index, index_update
from jax import jit, partial
from qutip_wrapper import rho_xy

jax.config.update('jax_enable_x64', True)

q_0 = qt.basis(2, 0)
q_1 = qt.basis(2, 1)

qubit_projectors = [q_0 * q_0.dag(), q_1 * q_1.dag(), q_1 * q_0.dag(), q_0 * q_1.dag()]
bell_states = [qt.bell_state(state="00"), qt.bell_state(state="01"), qt.bell_state(state="10"),
               qt.bell_state(state="11")]
proj = []
for b_i in bell_states:
    for b_j in bell_states:
        proj.append(0.5 * b_i * b_j.dag())


# print(pr00 + pr01 + pr10 + pr11)
Id_2 = qt.tensor([qt.qeye(2)] * 2)

proj_1 = jnp.array(qt.tensor(Id_2, qt.tensor(q_0, q_0) * qt.tensor(q_0, q_0).dag()).full(), dtype=jnp.complex128)
proj_2 = jnp.array(qt.tensor(Id_2, qt.tensor(q_1, q_0) * qt.tensor(q_1, q_0).dag()).full(), dtype=jnp.complex128)
proj_3 = jnp.array(qt.tensor(Id_2, qt.tensor(q_0, q_1) * qt.tensor(q_0, q_1).dag()).full(), dtype=jnp.complex128)
proj_4 = jnp.array(qt.tensor(Id_2, qt.tensor(q_1, q_1) * qt.tensor(q_1, q_1).dag()).full(), dtype=jnp.complex128)


def get_generators():
    clifford_set = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    generators = []
    for sigma_1 in clifford_set:
        for sigma_2 in clifford_set:
            generators.append(jnp.array(qt.tensor(sigma_2, sigma_1).full()))

    del (generators[0])

    return generators


generators = get_generators()


def S_n(i=0, j=1):
    s = np.identity(4)
    s[i, i] = 0
    s[j, j] = 0
    s[i, j] = 1
    s[j, i] = 1
    return jnp.array(s, dtype=np.complex128)


def tensor(qubit_list):
    s = None

    for i_el, el in enumerate(qubit_list):
        if i_el == 0:
            s = el
        else:
            s = jnp.kron(s, el)

    return s


def concurrence(rho: np.ndarray):
    Y = jnp.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    YY = jnp.kron(Y, Y)
    rho_tilde = (YY.dot(rho.T)).dot(YY)
    eigs = -jnp.sort(-jnp.linalg.eigvals(rho.dot(rho_tilde)))
    comp_arr = jnp.array([0., jnp.sqrt(eigs[0]) - jnp.sum(jnp.sqrt(eigs[1:]))])

    return jnp.max(jnp.real(comp_arr))


def partial_trace(rho: np.ndarray, subscripts='ijklmnkl->ijmn'):
    rho_rs = jnp.reshape(rho, np.array([2, 2, 2, 2, 2, 2, 2, 2]))
    # print(rho_rs.shape)
    rho_pt = jnp.einsum(subscripts, rho_rs).reshape(4, 4)

    return rho_pt


def test_partial_trace():
    dm = qt.tensor(rho_xy(0.25, 0.25, 1)[0], rho_xy(0.25, 0.25, 1)[0])
    dm_np = qt.tensor(rho_xy(0.25, 0.25, 1)[0], rho_xy(0.25, 0.25, 1)[0]).full()
    # print(dm_np.shape)
    dm_pt = dm.ptrace([2, 3])
    dm_np_pt = partial_trace(dm_np, subscripts='ijklmnkl->ijmn')

    # print(dm_np_pt.shape)
    # print(concurrence(dm_np_pt))

    return dm, dm_np


def general_U(s: np.ndarray):
    o = tensor([np.zeros((2, 2))] * 2)
    for i in range(s.shape[0]):
        o += 1j * s[i] * generators[i]

    return jsp.linalg.expm(o)


def m1_A1A2(s: np.ndarray):
    Id_4 = tensor([jnp.eye(2, dtype=jnp.complex128)] * 2)
    m = tensor([Id_4, general_U(s)])
    perm = tensor([tensor([jnp.eye(2, dtype=jnp.complex128), S_n(1, 2)]), jnp.eye(2, dtype=jnp.complex128)])
    m = (perm.dot(m)).dot(perm.T)
    return m


def m1_B1B2(s):
    Id_4 = tensor([jnp.eye(2, dtype=jnp.complex128)] * 2)
    m = tensor([general_U(s), Id_4])
    perm = tensor([tensor([jnp.eye(2, dtype=jnp.complex128), S_n(1, 2)]), jnp.eye(2, dtype=jnp.complex128)])
    m = (perm.dot(m)).dot(perm.T)
    return m


def m1(s: np.ndarray):
    m_tot = m1_A1A2(s[:15]).dot(m1_B1B2(s[:15]))

    return m_tot


def cost(rho_start: np.ndarray, x: np.ndarray, n_iter):
    # print(x)
    # assert x.shape[0] % n_iter == 0
    n_op_params = 15  # int((x.shape[0] / n_iter))

    # print(rho_start)
    rho_state = rho_start
    c_init = concurrence(rho_state)

    p0 = 1
    proj_set = (proj_1, proj_2, proj_3, proj_4)
    p_set = jnp.array([1] * (n_iter + 2))
    rho_set = jnp.array([jnp.zeros_like(rho_state)] * (n_iter + 2), dtype=jnp.complex128)

    p_set = index_update(p_set, index[0], p0)
    rho_set = index_update(rho_set, index[0], rho_state)
    # print(rho_set)

    for l in range(n_iter + 1):

        m = m1(x[: n_op_params])

        rho_0 = tensor([rho_set[l], rho_set[l]])
        rho_1 = (m.dot(rho_0)).dot(m.T.conjugate()) / (jnp.trace((m.T.conjugate().dot(m)).dot(rho_0)) + 1e-8)
        c_set = []
        rho_set = []
        prob_set = []

        for i_proj, proj in enumerate(proj_set):
            # print(l, i_p)
            prob = jnp.trace(proj.dot(proj.T.conjugate()).dot(rho_1))
            rho_new_ip = proj.dot(rho_1).dot(proj.T.conjugate()) / (
                    jnp.trace(proj.dot(proj.T.conjugate()).dot(rho_1)) + 1e-8)
            # print(rho_new_ip.shape)
            rho_new_ip_tilde = partial_trace(rho_new_ip)
            # Check concurrence

            c = concurrence(rho_new_ip_tilde)
            c_set.append(jnp.real(c))
            rho_set.append(rho_new_ip_tilde)
            prob_set.append(jnp.real(prob))

        c_set = jnp.array(c_set)
        idx = jnp.argmax(c_set)
        p_set = index_update(p_set, index[l + 1], jnp.asarray(prob_set)[idx])
        rho_set = index_update(rho_set, index[l + 1], jnp.asarray(rho_set, dtype=jnp.complex128)[idx])

    p_last_lvl = p_set[-1]
    rho_last_lvl = rho_set[-1]
    # idx = jnp.argmax(p_last_lvl)
    # rho_new = rho_last_lvl[idx]
    # p = p_last_lvl[idx]

    cost_value = 1 - concurrence(rho_last_lvl)
    # print(cost_value)
    # print(rho_start.eigenenergies())
    # print(rho_last_lvl.eigenenergies())
    # print(c_init)
    # print(qt.concurrence(rho_last_lvl))

    return cost_value, p_last_lvl, rho_last_lvl


def probability(rho_start: qt.Qobj, M, c: np.ndarray):
    p = cost(rho_start, c, M)[2]
    return p


def infidelity(rho_start: qt.Qobj, M, c: np.ndarray):
    cost_value = cost(rho_start, c, M)[0]
    return cost_value
