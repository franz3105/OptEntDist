import jax
import numpy as np
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

v1 = jnp.array([0, -1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0], dtype=jnp.complex128)
v2 = np.array([0, 1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0], dtype=np.complex128)
v3 = jnp.array([-1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)], dtype=jnp.complex128)
v4 = np.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)], dtype=np.complex128)

pr11 = jnp.outer(v1, v1)
pr22 = jnp.outer(v2, v2)
pr33 = jnp.outer(v3, v3)
pr44 = jnp.outer(v4, v4)

def rho_zsolt(c: np.float64):
    """

    :param c: parameter in [0,1) of the state that can be purified in 1 step
    :return: density matrix rho
    """
    gamma_sep = (1 / (np.sqrt(2))) * (1j * pr33 + pr22)
    rho = c * pr11 + (1 - c) * jnp.dot(gamma_sep, gamma_sep.T.conjugate())
    rho = rho / jnp.trace(rho)

    return rho

def rho_xy(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    dm = jnp.array([[(1 - x) / 4, (1j * y) / 4, -(1j * y) / 4, 1 / 4 * (-1 + x)],
                    [-(1j * y) / 4, (1 + x) / 4, 1 / 4 * (-1 - x), (1j * y) / 4],
                    [(1j * y) / 4, 1 / 4 * (-1 - x), (1 + x) / 4, -(1j * y) / 4],
                    [1 / 4 * (-1 + x), -(1j * y) / 4, (1j * y) / 4, (1 - x) / 4]], jnp.complex128)

    return dm

def tranformed_mau_state(x: np.float64):
    """

    :param x:
    :return:
    """

    trf_mau = jnp.array([[x / 2, 0, 0, -x / 2], [0, 0, 0, 0], [0, 0, 1 - x, 0], [-x / 2, 0, 0, x / 2]], np.complex128)
    return trf_mau

def werner(F: np.float64):
    """

    :param F: parameter in [0,1) of the Werner state
    :return: density matrix rho_werner
    """
    rho_werner = F * pr11 + ((1 - F) / 3) * pr22 + ((1 - F) / 3) * pr33 + ((1 - F) / 3) * pr44

    return rho_werner


def transformed_werner(x: np.float64):
    """

    :param x: Transformed Werner state
    :return: density matrix of the transformed Werner state
    """

    trf_w = (1 / 6) * jnp.array(
        [[1 + 2 * x, 0, 0, 1 - 4 * x], [0, 2 - 2 * x, 0, 0], [0, 0, 2 - 2 * x, 0], [1 - 4 * x, 0, 0, 1 + 2 * x]],
        np.complex128)
    return trf_w



