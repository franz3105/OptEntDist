import jax
import jax.numpy as jnp
import pandas as pd
import os

def tr_map(x):
    """
    Compute the trace of a list of matrices.
    :param x: List of matrices.
    :return: List of traces.
    """
    return jax.vmap(lambda a: jnp.trace(a))(x)


def load_data(dm_filename, a_vectors_filename, N=int(1e6), a_size=30):
    """
    Load data from files.
    :param dm_filename: File name for density matrices.
    :param a_vectors_filename: File name for a vectors.
    :param N: Number of samples.
    :param a_size: Dimension of a vectors.
    :return:
    """
    dm_N = pd.read_csv(dm_filename, dtype='float64')
    a_N = pd.read_csv(a_vectors_filename, dtype='float64')
    dm_N_real_imag = dm_N.values.reshape(N, 4, 4, 2)
    dm_N_complex = dm_N_real_imag[:, :, :, 0] + 1j * dm_N_real_imag[:, :, :, 1]
    a_N_val = a_N.values.reshape(N, a_size)
    n = 2
    d = 2 ** n
    return dm_N_complex.reshape(N, d, d), a_N_val


if __name__ == '__main__':
    dm_list = []
    n_dm_files = 10
    n_samples = int(1e6)
    cwd = os.getcwd()
    fld = os.path.join(cwd, "data_all_dms")

    for i in range(1, 1 + n_dm_files):
        dm_loaded = load_data(os.path.join(fld, f"random_dm_2_{i}_{n_samples}.csv"),
                              os.path.join(fld, f"random_a_vectors_{i}_{n_samples}.csv"), n_samples)

    # print(tr_map(dm_loaded))
