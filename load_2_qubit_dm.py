import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def tr_map(x):
    return jax.vmap(lambda a: jnp.trace(a))(x)


def load_data(dm_filename, a_vectors_filename):
    dm_N = pd.read_csv(dm_filename, dtype='float64')
    a_N = pd.read_csv(a_vectors_filename, dtype='float64')
    N = int(1e6)
    dm_N_real_imag = dm_N.values.reshape(N, 4, 4, 2)
    dm_N_complex = dm_N_real_imag[:, :, :, 0] + 1j * dm_N_real_imag[:, :, :, 1]
    a_N_val = a_N.values.reshape(N, 32)
    n = 2
    d = 2 ** n
    return dm_N_complex.reshape(N, d, d), a_N_val


if __name__ == '__main__':
    dm_loaded = load_data("random_dm_2.csv", "random_a_vectors.csv")
    #print(tr_map(dm_loaded))
