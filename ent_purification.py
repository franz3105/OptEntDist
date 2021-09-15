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


def m3_A1A2(s: np.ndarray) -> qt.Qobj:
    m = s[0] * projectors_A1A2[0] + \
        s[1] * projectors_A1A2[1] + \
        s[2] * projectors_A1A2[2] + \
        s[3] * projectors_A1A2[3]
    return m


def m3_B1B2(s) -> qt.Qobj:
    m = s[0] * projectors_B1B2[0] + \
        s[1] * projectors_B1B2[1] + \
        s[2] * projectors_B1B2[2] + \
        s[3] * projectors_B1B2[3]
    return qt.tensor(m)


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


def m3(s: np.ndarray) -> qt.Qobj:
    m_tot = m3_A1A2(s[:4]) * m3_B1B2(s[:4])

    return m_tot


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


def cost(rho_start: qt.Qobj, x: np.ndarray, M, n_iter):
    #assert x.shape[0] % n_iter == 0
    n_op_params = 4  # int((x.shape[0] / n_iter))

    # print(rho_start)
    rho_init = qt.tensor(rho_start, rho_start)
    # print(qt.concurrence(qt.ptrace(rho_init, [0, 1])))
    # print(rho_init.eigenenergies())
    # print(qt.concurrence(rho_start))
    p0 = 1
    proj_set = (proj_1, proj_2)
    p_set = [[1, ] * 2 ** i_lvl for i_lvl in range(n_iter+2)]
    rho_set = [[None, ] * 2 ** i_lvl for i_lvl in range(n_iter+2)]

    p_set[0][0] = p0
    rho_set[0][0] = rho_init

    for l in range(n_iter+1):
        #print(l)
        m = M(x[l * n_op_params:(l+1) * n_op_params])

        for i_p, p in enumerate(p_set[l]):
            #print(i_p)

            for i_proj, proj in enumerate(proj_set):
                #print(l, i_p)
                rho_0 = rho_set[l][i_p]
                rho_1 = m * rho_0 * m.dag() / ((m.dag() * m * rho_0).tr() + 1e-8)
                assert np.round(rho_1.tr(), 2) == 1.0
                prob = (proj * proj.dag() * rho_1).tr()
                rho_new_ip = proj * rho_1 * proj.dag() / ((proj * proj.dag() * rho_1).tr() + 1e-8)

                # print(p_1)
                # print(p_1+p_2+p_3+p_4)
                # print(np.round(np.array([p_1, p_2, p_3, p_4]), 2))
                # idx=0
                #print(rho_set)
                #print(prob)
                p_set[l+1][2*i_p + i_proj] = prob*p_set[l][i_p]
                rho_set[l+1][2*i_p + i_proj] = rho_new_ip
                #print(p_set)

                # print(p_set)
                if prob > 1 or prob < 0:
                    raise ValueError("Probabilities cannot lie outside range [0,1]")
                # print(rho_1.tr())
        #print(p_set)

        # rho_tilde_1 = qt.bel        pool = mp.Pool(NUM_CORES)
        #         actions = pool.map(self.single_step, (arg for arg in zip(self.envs, actions)))
        #         pool.close()
        #         pool.join()l_state(state="11").dag() * qt.ptrace(rho_new, [0, 1]) * qt.bell_state(state="11")
        # if np.round(1 - rho_tilde_1[0, 0] * p_1, 2) == 0:
        # if np.round(np.abs(3/2. - rho_tilde_1[0, 0] - p_1),2) == 0.0:
        #    print(p_1)
        #    print(p_2)
        #    print(rho_tilde_1[0,0])
        # if np.round(np.abs(1 - rho_tilde_1[0, 0]), 2) == 0:
        # print(f"Fidelity: {np.abs(rho_tilde_1[0, 0])}")
        # print(f"Probabiltiy {p_1}")

    p_last_lvl = np.array(p_set[-1])
    idx = np.argmax(p_last_lvl) - 1
    rho_new = rho_set[-1][idx]
    p = p_last_lvl[idx]

    #print(rho_new)
    concurrence = qt.concurrence(qt.ptrace(rho_new, [0, 1]))
    # fidelity = np.abs(rho_tilde_1[0, 0])
    # print(infidelity)
    cost_value = 1 - concurrence
    #print(cost_value)
    # if cost_value > 1:
    #    print(fidelity)
    #    print(p_1)

    return cost_value, p


def probability(rho_start: qt.Qobj, M, c: np.ndarray):
    p = cost(rho_start, c, M)[2]
    return p


def infidelity(rho_start: qt.Qobj, M, c: np.ndarray):
    cost_value = cost(rho_start, c, M)[0]
    return cost_value
