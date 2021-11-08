import qutip as qt
import numpy as np
import jax.numpy as jnp

q_0 = qt.basis(2, 0)
q_1 = qt.basis(2, 1)
pr00 = qt.bell_state(state="00") * qt.bell_state(state="00").dag()
pr01 = qt.bell_state(state="01") * qt.bell_state(state="01").dag()
pr10 = qt.bell_state(state="10") * qt.bell_state(state="10").dag()
pr11 = qt.bell_state(state="11") * qt.bell_state(state="11").dag()


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


def Zloc(num_qubits, theta, i):
    out = - 1.j * theta * qt.sigmaz() / 2
    qubit_list = [qt.qeye(2)] * num_qubits
    qubit_list[i] = out.expm()
    out = qt.tensor(qubit_list)
    return out


def werner(F: float):
    rho_werner = F * pr11 + ((1 - F) / 3) * pr00 + ((1 - F) / 3) * pr10 + ((1 - F) / 3) * pr01
    return rho_werner


def rho_xy_phi(eta_damp, phi, n_photon):
    # print(eta_damp, phi, n_photon)
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
