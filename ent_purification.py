# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import qutip as qt
from qutip.measurement import measure, measurement_statistics
from qutip.qip.operations import controlled_gate, cnot, molmer_sorensen
import numpy as np
import matplotlib.pyplot as plt
import random

q_0 = qt.basis(2, 0)
q_1 = qt.basis(2, 1)

qubit_projectors = [q_0 * q_0.dag(), q_1 * q_1.dag(), q_1 * q_0.dag(), q_0 * q_1.dag()]
bell_states = [qt.bell_state(state="00"), qt.bell_state(state="01"), qt.bell_state(state="10"),
               qt.bell_state(state="11")]
proj = []
for b_i in bell_states:
    for b_j in bell_states:
        proj.append(0.5 * b_i * b_j.dag())

# print(proj)
s = qt.tensor(qt.qzero(2), qt.qzero(2))
print(len(proj))
for p in proj:
    s += p

print(s)
pr00 = qt.bell_state(state="00") * qt.bell_state(state="00").dag()
pr01 = qt.bell_state(state="01") * qt.bell_state(state="01").dag()
pr10 = qt.bell_state(state="10") * qt.bell_state(state="10").dag()
pr11 = qt.bell_state(state="11") * qt.bell_state(state="11").dag()
print(pr00 + pr01 + pr10 + pr11)
Id_2 = qt.tensor([qt.qeye(2)] * 2)

proj_1 = qt.tensor(Id_2, qt.tensor(q_0, q_0) * qt.tensor(q_0, q_0).dag())
proj_2 = qt.tensor(Id_2, qt.tensor(q_1, q_0) * qt.tensor(q_1, q_0).dag())
proj_3 = qt.tensor(Id_2, qt.tensor(q_0, q_1) * qt.tensor(q_0, q_1).dag())
proj_4 = qt.tensor(Id_2, qt.tensor(q_1, q_1) * qt.tensor(q_1, q_1).dag())

clifford_set = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
generators = []
for sigma_1 in clifford_set:
    for sigma_2 in clifford_set:
        generators.append(qt.tensor(sigma_2, sigma_1))

del (generators[0])
# print(proj_1 + proj_2 + proj_3 + proj_4)
# print(qt.bell_state(state="00"))
# print(qt.bell_state(state="01"))
# print(qt.bell_state(state="10"))
# print(qt.bell_state(state="11"))

# print(pr11 * pr11.dag())# + pr11 * pr11.dag() + pr01 * pr01.dag() + pr10 * pr10.dag())
p_length = 100


def S_n(i=0, j=1):
    s = np.identity(4)
    s[i, i] = 0
    s[j, j] = 0
    s[i, j] = 1
    s[j, i] = 1
    return qt.Qobj(s, dims=[[2, ] * 2, [2, ] * 2])


# print(S_n(i=1, j=2))


def bell_projectors_B1B2():
    proj1 = 0.5 * (qt.tensor(qt.qeye(2), q_0 * q_0.dag(), qt.qeye(2), q_1 * q_1.dag()) +
                   qt.tensor(qt.qeye(2), q_1 * q_1.dag(), qt.qeye(2), q_0 * q_0.dag()) -
                   qt.tensor(qt.qeye(2), q_0 * q_1.dag(), qt.qeye(2), q_1 * q_0.dag()) -
                   qt.tensor(qt.qeye(2), q_1 * q_0.dag(), qt.qeye(2), q_0 * q_1.dag()))

    proj2 = 0.5 * (qt.tensor(qt.qeye(2), q_0 * q_0.dag(), qt.qeye(2), q_1 * q_1.dag()) +
                   qt.tensor(qt.qeye(2), q_1 * q_1.dag(), qt.qeye(2), q_0 * q_0.dag()) +
                   qt.tensor(qt.qeye(2), q_0 * q_1.dag(), qt.qeye(2), q_1 * q_0.dag()) +
                   qt.tensor(qt.qeye(2), q_1 * q_0.dag(), qt.qeye(2), q_0 * q_1.dag()))

    proj3 = 0.5 * (qt.tensor(qt.qeye(2), q_0 * q_0.dag(), qt.qeye(2), q_0 * q_0.dag()) +
                   qt.tensor(qt.qeye(2), q_1 * q_1.dag(), qt.qeye(2), q_1 * q_1.dag()) +
                   qt.tensor(qt.qeye(2), q_0 * q_1.dag(), qt.qeye(2), q_0 * q_1.dag()) +
                   qt.tensor(qt.qeye(2), q_1 * q_0.dag(), qt.qeye(2), q_1 * q_0.dag()))

    proj4 = 0.5 * (qt.tensor(qt.qeye(2), q_0 * q_0.dag(), qt.qeye(2), q_0 * q_0.dag()) +
                   qt.tensor(qt.qeye(2), q_1 * q_1.dag(), qt.qeye(2), q_1 * q_1.dag()) -
                   qt.tensor(qt.qeye(2), q_0 * q_1.dag(), qt.qeye(2), q_0 * q_1.dag()) -
                   qt.tensor(qt.qeye(2), q_1 * q_0.dag(), qt.qeye(2), q_1 * q_0.dag()))
    # print(proj_1 + proj_2 + proj_3 + proj_4)
    return proj1, proj2, proj3, proj4


def bell_projectors_A1A2():
    proj1 = 0.5 * (qt.tensor(q_0 * q_0.dag(), qt.qeye(2), q_1 * q_1.dag(), qt.qeye(2)) +
                   qt.tensor(q_1 * q_1.dag(), qt.qeye(2), q_0 * q_0.dag(), qt.qeye(2)) -
                   qt.tensor(q_0 * q_1.dag(), qt.qeye(2), q_1 * q_0.dag(), qt.qeye(2)) -
                   qt.tensor(q_1 * q_0.dag(), qt.qeye(2), q_0 * q_1.dag(), qt.qeye(2)))

    proj2 = 0.5 * (qt.tensor(q_0 * q_0.dag(), qt.qeye(2), q_1 * q_1.dag(), qt.qeye(2)) +
                   qt.tensor(q_1 * q_1.dag(), qt.qeye(2), q_0 * q_0.dag(), qt.qeye(2)) +
                   qt.tensor(q_0 * q_1.dag(), qt.qeye(2), q_1 * q_0.dag(), qt.qeye(2)) +
                   qt.tensor(q_1 * q_0.dag(), qt.qeye(2), q_0 * q_1.dag(), qt.qeye(2)))

    proj3 = 0.5 * (qt.tensor(q_0 * q_0.dag(), qt.qeye(2), q_0 * q_0.dag(), qt.qeye(2)) +
                   qt.tensor(q_1 * q_1.dag(), qt.qeye(2), q_1 * q_1.dag(), qt.qeye(2)) +
                   qt.tensor(q_0 * q_1.dag(), qt.qeye(2), q_0 * q_1.dag(), qt.qeye(2)) +
                   qt.tensor(q_1 * q_0.dag(), qt.qeye(2), q_1 * q_0.dag(), qt.qeye(2)))

    proj4 = 0.5 * (qt.tensor(q_0 * q_0.dag(), qt.qeye(2), q_0 * q_0.dag(), qt.qeye(2)) +
                   qt.tensor(q_1 * q_1.dag(), qt.qeye(2), q_1 * q_1.dag(), qt.qeye(2)) -
                   qt.tensor(q_0 * q_1.dag(), qt.qeye(2), q_0 * q_1.dag(), qt.qeye(2)) -
                   qt.tensor(q_1 * q_0.dag(), qt.qeye(2), q_1 * q_0.dag(), qt.qeye(2)))

    # print(proj_1 + proj_2 + proj_3 + proj_4)
    return proj1, proj2, proj3, proj4


def phased_bell_state(phi: float, state="+"):
    if state == "+":
        out = (qt.tensor(q_0, q_0) * np.exp(-1j * phi) + qt.tensor(q_1, q_1) * np.exp(1j * phi)) \
              / np.sqrt(2)
    elif state == "-":
        out = (qt.tensor(q_0, q_0) * np.exp(-1j * phi) - qt.tensor(q_1, q_1) * np.exp(1j * phi)) \
              / np.sqrt(2)
    else:
        raise NotImplementedError("Only + and - are valid states")

    return out


def rho_zsolt(c: float):
    gamma_sep = (1 / (np.sqrt(2))) * (1j * qt.bell_state(state="01") + qt.bell_state(state="10"))
    rho = c * pr11 + (1 - c) * gamma_sep * gamma_sep.dag()
    return rho / rho.tr()


def rho_xy_phi(eta_damp, phi, n_photon):
    print(eta_damp, phi, n_photon)
    x = np.exp(- n_photon * (1 - np.cos(2 * phi)) * (1 - eta_damp)) * np.cos(
        n_photon * np.sin(2 * phi) * (1 - eta_damp))
    y = np.exp(- n_photon * (1 - np.cos(2 * phi)) * (1 - eta_damp)) * np.sin(
        n_photon * np.sin(2 * phi) * (1 - eta_damp))

    mix_phi = phased_bell_state(phi, state="-") * qt.bell_state(state="11").dag()

    rho_out = (1 + x) * pr11 + (1 - x) * phased_bell_state(phi, state="-") * phased_bell_state(phi,
                                                                                               state="-").dag() + \
              1j * y * (mix_phi - mix_phi.dag())

    return rho_out / rho_out.tr(), x, y


def rho_xy(x, y, phase=np.pi / 2):
    # print(x, y, phase)
    mix_phi = phased_bell_state(phase, state="-") * qt.bell_state(state="11").dag()

    rho_out = (1 + x) * pr11 + (1 - x) * phased_bell_state(phase, state="-") * phased_bell_state(phase,
                                                                                                 state="-").dag() + \
              1j * y * (mix_phi - mix_phi.dag())

    # print(f"State: {rho_out}")
    # print(pr11)
    # print(rho_out.tr())
    return rho_out / 2., x, y


projectors_A1A2 = bell_projectors_A1A2()
projectors_B1B2 = bell_projectors_B1B2()


def MS(num_qubits, theta, phi):
    Sy = qt.tensor([qt.qzero(2)] * num_qubits)
    Sx = qt.tensor([qt.qzero(2)] * num_qubits)
    qubit_list_x = [qt.qeye(2)] * num_qubits
    qubit_list_y = [qt.qeye(2)] * num_qubits
    for i in range(num_qubits):
        qubit_list_x[i] = qt.sigmax()
        qubit_list_y[i] = qt.sigmay()
        Sx += qt.tensor(qubit_list_x)
        Sy += qt.tensor(qubit_list_y)
        qubit_list_x = [qt.qeye(2)] * num_qubits
        qubit_list_y = [qt.qeye(2)] * num_qubits

    out = - 1.j * theta * (Sx * np.cos(phi) + Sy * np.sin(phi)) ** 2 / 4
    out = out.expm()
    return out


def proj_U(s: np.ndarray):
    u = np.exp(1j * s[0]) * pr00 + np.exp(1j * s[1]) * pr01 + np.exp(1j * s[2]) * pr10 + np.exp(1j * s[3]) * pr11
    # print(u)
    # print(u.shape)

    return u


def general_U(s: np.ndarray):
    o = qt.tensor([qt.qzero(2)] * 2)
    for i in range(s.shape[0]):
        o += 1j*s[i] * generators[i]

    return o.expm()


def Zloc(num_qubits, theta, i):
    out = - 1.j * theta * qt.sigmaz() / 2
    qubit_list = [qt.qeye(2)] * num_qubits
    qubit_list[i] = out.expm()
    out = qt.tensor(qubit_list)
    return out


def m2_B1B2(s: np.ndarray) -> qt.Qobj:
    Id_4 = qt.tensor([qt.qeye(2)] * 2)
    m = qt.tensor(Id_4, Zloc(2, s[3], 1) * MS(2, s[0], s[1]) * Zloc(2, s[2], 1))
    perm = qt.tensor(qt.tensor(qt.qeye(2), S_n(1, 2)), qt.qeye(2))
    m = perm * m * perm.dag()
    return m


def m2_A1A2(s: np.ndarray) -> qt.Qobj:
    Id_4 = qt.tensor([qt.qeye(2)] * 2)
    m = qt.tensor(Zloc(2, s[3], 1) * MS(2, s[0], s[1]) * Zloc(2, s[2], 1), Id_4)
    perm = qt.tensor(qt.tensor(qt.qeye(2), S_n(1, 2)), qt.qeye(2))
    m = perm * m * perm.dag()
    return m


def m4_A1A2(s: np.ndarray):
    Id_4 = qt.tensor([qt.qeye(2)] * 2)
    m = qt.tensor(Id_4, qt.cnot())
    perm = qt.tensor(qt.tensor(qt.qeye(2), S_n(1, 2)), qt.qeye(2))
    m = perm * m * perm.dag()
    return m


def m4_B1B2(s: np.ndarray) -> qt.Qobj:
    Id_4 = qt.tensor([qt.qeye(2)] * 2)
    m = qt.tensor(qt.cnot(), Id_4)
    perm = qt.tensor(qt.tensor(qt.qeye(2), S_n(1, 2)), qt.qeye(2))
    m = perm * m * perm.dag()
    return m


def m3_A1A2(s: np.ndarray) -> qt.Qobj:
    m = s[0] * projectors_A1A2[0] + \
        s[1] * projectors_A1A2[1] + \
        s[2] * projectors_A1A2[2] + \
        s[3] * projectors_A1A2[3]

    return m / np.linalg.norm(s)


def m3_B1B2(s) -> qt.Qobj:
    m = s[0] * projectors_B1B2[0] + \
        s[1] * projectors_B1B2[1] + \
        s[2] * projectors_B1B2[2] + \
        s[3] * projectors_B1B2[3]

    return qt.tensor(m) / np.linalg.norm(s)


def m1_A1A2(s: np.ndarray) -> qt.Qobj:
    Id_4 = qt.tensor([qt.qeye(2)] * 2)
    m = qt.tensor(Id_4, general_U(s))
    perm = qt.tensor(qt.tensor(qt.qeye(2), S_n(1, 2)), qt.qeye(2))
    m = perm * m * perm.dag()
    return qt.tensor(m)


def m1_B1B2(s) -> qt.Qobj:
    Id_4 = qt.tensor([qt.qeye(2)] * 2)
    m = qt.tensor(general_U(s), Id_4)
    perm = qt.tensor(qt.tensor(qt.qeye(2), S_n(1, 2)), qt.qeye(2))
    m = perm * m * perm.dag()
    return qt.tensor(m)


def m3(s: np.ndarray) -> qt.Qobj:
    m_tot = m3_A1A2(s[:4]) * m3_B1B2(s[:4])

    return m_tot


def m4(s: np.ndarray) -> qt.Qobj:
    m_tot = m4_A1A2(s[:4]) * m4_B1B2(s[:4])

    return m_tot


def m1(s: np.ndarray) -> qt.Qobj:
    m_tot = m1_A1A2(s[:15]) * m1_B1B2(s[:15])
    # print(m_tot * m_tot.dag())
    #    m1_A1A2(s[:4]) * m1_B1B2(s[4:]) - m1_A1A2(s[:4]) * m1_B1B2(s[4:])) == 0
    return m_tot


def m2(s: np.ndarray) -> qt.Qobj:
    return m2_A1A2(s[:4]) * m2_B1B2(s[:4])


def werner(F: float):
    rho_werner = F * pr11 + ((1 - F) / 3) * pr00 + ((1 - F) / 3) * pr10 + ((1 - F) / 3) * pr01
    return rho_werner


def cost(rho_start: qt.Qobj, x: np.ndarray, M, n_iter):
    # print(x)
    # assert x.shape[0] % n_iter == 0
    n_op_params = 15  # int((x.shape[0] / n_iter))

    # print(rho_start)
    rho_state = rho_start
    c_init = qt.concurrence(rho_state)

    p0 = 1
    proj_set = (proj_1, proj_2, proj_3, proj_4)
    p_set = [1] * (n_iter + 2)
    rho_set = [None] * (n_iter + 2)

    p_set[0] = p0
    rho_set[0] = rho_state
    # print(rho_set)

    for l in range(n_iter + 1):
        # print(l)
        u = x[:n_op_params]  # + 1j * x[n_op_params:2 * n_op_params]
        # for i in range(n_op_params):
        #    u[i] = u[i] / np.abs(u[i])
        # print(u[0]*u[0].conjugate())
        # u = 4*u / np.linalg.norm(u)
        # print(u)
        m = M(u[: n_op_params])
        # print(m * m.dag())
        # print(u)
        #print(m)
        # print(x[l * n_op_params:(l + 1) * n_op_params])
        # print(i_p)
        rho_0 = qt.tensor(rho_set[l], rho_set[l])
        rho_1 = m * rho_0 * m.dag() / ((m.dag() * m * rho_0).tr() + 1e-8)
        # print((rho_1 - rho_0).full().round(2))
        c_set = []
        rho_set = []
        prob_set = []

        for i_proj, proj in enumerate(proj_set):
            # print(l, i_p)
            prob = (proj * proj.dag() * rho_1).tr()
            rho_new_ip = proj * rho_1 * proj.dag() / ((proj * proj.dag() * rho_1).tr() + 1e-8)
            rho_new_ip_tilde = qt.ptrace(rho_new_ip, [0, 1])
            # Check concurrence

            c = qt.concurrence(rho_new_ip_tilde)
            c_set.append(c)
            rho_set.append(rho_new_ip_tilde)
            prob_set.append(prob)

        c_set = np.array(c_set)
        idx = np.argmax(c_set)
        p_set[l + 1] = prob_set[idx]
        rho_set[l + 1] = rho_set[idx]

    p_last_lvl = p_set[-1]
    rho_last_lvl = rho_set[-1]
    # idx = np.argmax(p_last_lvl)
    # rho_new = rho_last_lvl[idx]
    # p = p_last_lvl[idx]

    c_set = []
    cost_value = 1 - qt.concurrence(rho_last_lvl)
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
