import jax
import numpy as np
import jax.numpy as jnp

v1 = jnp.array([0, -1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0], dtype=jnp.complex128)
v2 = np.array([0, 1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0], dtype=np.complex)
v3 = jnp.array([-1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)], dtype=jnp.complex128)
v4 = np.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)], dtype=np.complex)

pr11 = jnp.outer(v1, v1)
pr22 = jnp.outer(v2, v2)
pr33 = jnp.outer(v3, v3)
pr44 = jnp.outer(v4, v4)

jax.config.update('jax_enable_x64', True)


def rho_zsolt(c: float):
    gamma_sep = (1 / (np.sqrt(2))) * (1j * pr33 + pr22)
    rho = c * pr11 + (1 - c) * jnp.dot(gamma_sep, gamma_sep.T.conjugate())
    print(rho.shape)
    return rho / jnp.trace(rho)


def werner(F: float):
    rho_werner = F * pr11 + ((1 - F) / 3) * pr22 + ((1 - F) / 3) * pr33 + ((1 - F) / 3) * pr44
    return rho_werner
