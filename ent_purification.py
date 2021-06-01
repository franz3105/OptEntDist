# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import qutip as qt
from qutip.qip.operations import controlled_gate, cnot, molmer_sorensen
import numpy as np
import matplotlib.pyplot as plt
import random

q_0 = qt.basis(2, 0)
q_1 = qt.basis(2, 1)

qubit_projectors = [q_0 * q_0.dag(), q_1 * q_1.dag(), q_1 * q_0.dag(), q_0 * q_1.dag()]
pr00 = qt.bell_state(state="00") * qt.bell_state(state="00").dag()
pr01 = qt.bell_state(state="01") * qt.bell_state(state="01").dag()
pr10 = qt.bell_state(state="10") * qt.bell_state(state="10").dag()
pr11 = qt.bell_state(state="11") * qt.bell_state(state="11").dag()
Id_2 = qt.tensor([qt.qeye(2)] * 2)

proj_1 = qt.tensor(Id_2, qt.tensor(q_0, q_0) * qt.tensor(q_0, q_0).dag())
proj_2 = qt.tensor(Id_2, qt.tensor(q_1, q_0) * qt.tensor(q_1, q_0).dag())
proj_3 = qt.tensor(Id_2, qt.tensor(q_0, q_1) * qt.tensor(q_0, q_1).dag())
proj_4 = qt.tensor(Id_2, qt.tensor(q_1, q_1) * qt.tensor(q_1, q_1).dag())

print(proj_1 + proj_2 + proj_3 + proj_4)
print(qt.bell_state(state="00"))
print(qt.bell_state(state="01"))
print(qt.bell_state(state="10"))
print(qt.bell_state(state="11"))

# print(pr11 * pr11.dag())# + pr11 * pr11.dag() + pr01 * pr01.dag() + pr10 * pr10.dag())
p_length = 100


def S_n(i=0, j=1):
    s = np.identity(4)
    s[i, i] = 0
    s[j, j] = 0
    s[i, j] = 1
    s[j, i] = 1
    return qt.Qobj(s, dims=[[2, ] * 2, [2, ] * 2])


print(S_n(i=1, j=2))


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
    print(proj_1 + proj_2 + proj_3 + proj_4)
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

    print(proj_1 + proj_2 + proj_3 + proj_4)
    return proj1, proj2, proj3, proj4


def phased_bell_state(phi: float, state="00"):
    if state == "00":
        out = (qt.bell_state(state=state) * np.exp(-1j * phi) + qt.bell_state(state=state) * np.exp(1j * phi)) \
              / np.sqrt(2)
    elif state == "11":
        out = (qt.bell_state(state=state) * np.exp(-1j * phi) - qt.bell_state(state=state) * np.exp(1j * phi)) \
              / np.sqrt(2)
    else:
        raise NotImplementedError("Only 00 and 11 are valid states")

    return out


def rho_zsolt(c: float):
    gamma_sep = (1 / (np.sqrt(2))) * (qt.bell_state(state="00") + qt.bell_state(state="10"))
    rho = c * pr01 + (1 - c) * gamma_sep * gamma_sep.dag()
    return rho


def rho_xy_phi(eta_damp, phi, n_photon):
    print(eta_damp, phi, n_photon)
    x = np.exp(- n_photon * (1 - np.cos(2 * phi)) * (1 - eta_damp)) * np.cos(
        n_photon * np.sin(2 * phi) * (1 - eta_damp))
    y = np.exp(- n_photon * (1 - np.cos(2 * phi)) * (1 - eta_damp)) * np.sin(
        n_photon * np.sin(2 * phi) * (1 - eta_damp))

    mix_phi = phased_bell_state(phi, state="11") * qt.bell_state(state="01").dag()

    rho_out = (1 + x) * pr01 + (1 - x) * phased_bell_state(phi, state="11") * phased_bell_state(phi,
                                                                                                state="11").dag() + \
              1j * y * (mix_phi - mix_phi.dag())

    return rho_out / rho_out.tr(), x, y


def rho_xy(x, y, phase=np.pi / 2):
    # print(x, y, phase)
    mix_phi = phased_bell_state(phase, state="11") * qt.bell_state(state="01").dag()

    rho_out = (1 + x) * pr01 + (1 - x) * phased_bell_state(phase, state="11") * phased_bell_state(phase,
                                                                                                  state="11").dag() + \
              1j * y * (mix_phi - mix_phi.dag())

    print(f"State: {rho_out}")
    return rho_out / 4., x, y


projectors_A1A2 = bell_projectors_A1A2()
projectors_B1B2 = bell_projectors_B1B2()


def MS(num_qubits, theta, phi):
    Sy = qt.tensor([qt.qzero(2)] * num_qubits)
    Sx = qt.tensor([qt.qzero(2)] * num_qubits)
    qubit_list_x = [qt.qeye(2)] * num_qubits
    qubit_list_y = [qt.qeye(2)] * num_qubits
    for i in range(num_qubits):
        qubit_list_x[i] = qt.sigmax()
        qubit_list_y[i] = qt.sigmax()
        Sx += qt.tensor(qubit_list_x)
        Sy += qt.tensor(qubit_list_y)
        qubit_list_x = [qt.qeye(2)] * num_qubits
        qubit_list_y = [qt.qeye(2)] * num_qubits

    out = - 1.j * theta * (Sx * np.cos(phi) + Sy * np.sin(phi)) ** 2 / 4
    out = out.expm()
    return out


def m2_B1B2(s: np.ndarray) -> qt.Qobj:
    Id_4 = qt.tensor([qt.qeye(2)] * 2)
    m = qt.tensor(Id_4, MS(2, s[0], s[1]))
    perm = qt.tensor(qt.tensor(qt.qeye(2), S_n(1, 2)), qt.qeye(2))
    m = perm * m * perm.dag()
    return m


def m2_A1A2(s: np.ndarray) -> qt.Qobj:
    Id_4 = qt.tensor([qt.qeye(2)] * 2)
    m = qt.tensor(MS(2, s[0], s[1]), Id_4)
    perm = qt.tensor(qt.tensor(qt.qeye(2), S_n(1, 2)), qt.qeye(2))
    m = perm * m * perm.dag()
    return m


def m1_A1A2(s: np.ndarray) -> qt.Qobj:
    m = np.exp(1j * s[0]) * projectors_A1A2[0] + \
        np.exp(1j * s[1]) * projectors_A1A2[1] + \
        np.exp(1j * s[2]) * projectors_A1A2[2] + \
        np.exp(1j * s[3]) * projectors_A1A2[3]
    return qt.tensor(m)


def m1_B1B2(s) -> qt.Qobj:
    m = np.exp(1j * s[0]) * projectors_B1B2[0] + \
        np.exp(1j * s[1]) * projectors_B1B2[1] + \
        np.exp(1j * s[2]) * projectors_B1B2[2] + \
        np.exp(1j * s[3]) * projectors_B1B2[3]
    return qt.tensor(m)


def m1(s: np.ndarray) -> qt.Qobj:
    m_tot = m1_A1A2(s[:4]) * m1_B1B2(s[:4])
    # print(m_tot * m_tot.dag())
    #    m1_A1A2(s[:4]) * m1_B1B2(s[4:]) - m1_A1A2(s[:4]) * m1_B1B2(s[4:])) == 0
    return m_tot


def m2(s: np.ndarray) -> qt.Qobj:
    return m2_A1A2(s[:2]) * m2_B1B2(s[2:])


def werner(F: float):
    rho_werner = F * pr11 + ((1 - F) / 3) * pr00 + ((1 - F) / 3) * pr10 + ((1 - F) / 3) * pr01
    return rho_werner


def cost(rho_start: qt.Qobj, c: np.ndarray, M, n_iter=5):
    x = c

    if M == m1:
        m = M(x)
    else:
        m = M(x)

    #print(rho_start)
    rho_init = qt.tensor(rho_start, rho_start)

    for l in range(n_iter):
        # print(rho_init.tr())
        rho_1 = m * rho_init * m.dag() / ((m.dag() * m * rho_init).tr() + 1e-8)
        assert np.round(rho_1.tr(), 2) == 1.0
        rho_new = proj_1 * rho_1 * proj_1.dag() / ((proj_1 * proj_1.dag() * rho_1).tr() + 1e-8)
        p_1 = (proj_1 * proj_1.dag() * rho_1).tr()
        # print(p_1)
        if p_1 > 1 or p_1 < 0:
            raise ValueError("Probabilities cannot lie outside range [0,1]")
        # print(rho_1.tr())
        p_2 = (proj_2 * proj_2.dag() * rho_1).tr()
        p_3 = (proj_3 * proj_3.dag() * rho_1).tr()
        p_4 = (proj_4 * proj_4.dag() * rho_1).tr()

        rho_tilde_1 = qt.bell_state(state="01").dag() * qt.ptrace(rho_new, [0, 1]) * qt.bell_state(state="01")
        # if np.round(1 - rho_tilde_1[0, 0] * p_1, 2) == 0:
        # if np.round(np.abs(3/2. - rho_tilde_1[0, 0] - p_1),2) == 0.0:
        #    print(p_1)
        #    print(p_2)
        #    print(rho_tilde_1[0,0])
        if np.round(np.abs(1 - rho_tilde_1[0, 0]), 2) == 0:
            print(f"Fidelity: {np.abs(rho_tilde_1[0, 0])}")
            print(f"Probabiltiy {p_1}")

        rho_init = rho_new

    return np.abs((1 - rho_tilde_1[0, 0]) / (p_1 + 1e-8)), np.abs(1 - rho_tilde_1[0, 0]), p_1


def purify(rho_start: qt.Qobj, M,  c: np.ndarray, n_iter=1):
    x = c

    if M == m1:
        m = M(x)
    else:
        m = M(x)
    #print(rho_start)
    rho_init = qt.tensor(rho_start, rho_start)

    for l in range(n_iter):
        # print(rho_init.tr())
        rho_1 = m * rho_init * m.dag() / ((m.dag() * m * rho_init).tr() + 1e-8)
        assert np.round(rho_1.tr(), 2) == 1.0
        rho_new = proj_1 * rho_1 * proj_1.dag() / ((proj_1 * proj_1.dag() * rho_1).tr() + 1e-8)
        p_1 = (proj_1 * proj_1.dag() * rho_1).tr()
        # print(p_1)
        if p_1 > 1 or p_1 < 0:
            raise ValueError("Probabilities cannot lie outside range [0,1]")
        # print(rho_1.tr())
        p_2 = (proj_2 * proj_2.dag() * rho_1).tr()
        p_3 = (proj_3 * proj_3.dag() * rho_1).tr()
        p_4 = (proj_4 * proj_4.dag() * rho_1).tr()

        rho_tilde_1 = qt.bell_state(state="01").dag() * qt.ptrace(rho_new, [0, 1]) * qt.bell_state(state="01")
        # if np.round(1 - rho_tilde_1[0, 0] * p_1, 2) == 0:
        # if np.round(np.abs(3/2. - rho_tilde_1[0, 0] - p_1),2) == 0.0:
        #    print(p_1)
        #    print(p_2)
        #    print(rho_tilde_1[0,0])
        if np.round(np.abs(1 - rho_tilde_1[0, 0]), 2) == 0:
            print(f"Fidelity: {np.abs(rho_tilde_1[0, 0])}")
            print(f"Probabiltiy {p_1}")

        rho_init = rho_new

    return rho_init


def probability(rho_start: qt.Qobj, M, c: np.ndarray,  n_iter=1):
    rho_1 = purify(rho_start, M,c, n_iter=n_iter)
    p_1 = (proj_1 * proj_1.dag() * rho_1).tr()
    p_2 = (proj_2 * proj_2.dag() * rho_1).tr()
    p_3 = (proj_3 * proj_3.dag() * rho_1).tr()
    p_4 = (proj_4 * proj_4.dag() * rho_1).tr()
    return p_1


def fidelity(rho_start: qt.Qobj, M, c: np.ndarray, n_iter=1):
    rho_1 = purify(rho_start, M, c, n_iter=n_iter)
    rho_tilde_1 = qt.bell_state(state="01").dag() * qt.ptrace(rho_1, [0, 1]) * qt.bell_state(state="01")
    return np.abs((1 - rho_tilde_1[0, 0]))
