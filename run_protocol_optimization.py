import numpy as np
import os
import json
import jax
import datetime
import pickle

from qutip_wrapper import werner, rho_zsolt, transformed_werner, tranformed_mau_state
from optimize_avg_c_protocol import werner_sample, mau_state_sample, xy_state_sample, xy_state_rot, optimize_protocol, \
    plot, concurrence
from load_2_qubit_dm import load_data


def run_simulation(state_names=None,
                   n_points=1,
                   n_iter=1,
                   n_guesses=1,
                   gate_name="SU4",
                   seed=0,
                   proj_vec=np.zeros(4),
                   n_opt=10):
    if state_names is None:
        state_names = ["2-qubit-dm"]

    jax.config.update('jax_platforms', 'cpu')
    jax.config.update("jax_debug_nans", True)

    cwd = os.getcwd()
    data_folder = os.path.join(cwd, "data")
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    else:
        pass

    # List of names of avaialble states
    # state_names = ["Werner"]

    if gate_name == "circuit":
        n_components = 24
    else:
        n_components = 30

    np.random.seed(seed)

    for sn in state_names:
        if sn == "2-qubit-dm":
            dm_start, points = load_data("random_dm_2.csv", "random_a_vectors.csv")
            n_points = dm_start.shape[0]
            # c_dm = jax.vmap(concurrence)(dm_start)
            # idx_non_zero = np.where(c_dm > 1e-15)[0]
            # dm_start = dm_start[idx_non_zero]
            # points = points[idx_non_zero]
            # print(dm_start.shape)
        elif sn == "Werner":
            dm_start, points = werner_sample(werner, n_points, 0.5, 1.0)
        elif sn == "Transformed Werner":
            dm_start, points = werner_sample(transformed_werner, n_points, 0.51, 0.99)
        elif sn == "Mauricio":
            dm_start, points = mau_state_sample(rho_zsolt, n_points, 0.51, 0.99)
        elif sn == "Transformed Mauricio":
            dm_start, points = mau_state_sample(tranformed_mau_state, n_points, 0.51, 0.99)
        elif sn == "XY_avg":
            dm_start, points = xy_state_sample(n_points, 1e-3, 0.999)
        elif sn == "XY_single":
            x_array = np.array([0.25])
            y_array = np.array([0.25])
            dm_start, points = xy_state_rot(x_array, y_array)
        else:
            raise NotImplementedError("This type of state is not implemented!")
        hyperparams = dict(state_name=sn, n_points=n_points, n_iter=n_iter, n_guesses=n_guesses,
                           n_components=n_components, gate_name=gate_name, seed=seed, proj_vector=proj_vec, n_opt=n_opt)

        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sn_folder_path = os.path.join(data_folder, f"{sn}" + "_" + time_str)
        if not os.path.exists(sn_folder_path):
            os.mkdir(sn_folder_path)
        else:
            pass

        best_concurrences, probabilities, best_unitaries, data_points, gtc_mean, gtp_mean, gtc_std, gtp_std = \
            optimize_protocol(dm_start, points, n_iter, n_guesses, n_components, gate_name=gate_name,
                              proj_vec=proj_vec, n_idx=n_opt)

        # np.savetxt(os.path.join(sn_folder_path, "plotlines.txt"), plot_lines)
        np.savetxt(os.path.join(sn_folder_path, f"best_concurrences_{0}.txt"), best_concurrences)
        np.savetxt(os.path.join(sn_folder_path, f"avg_probabilities_{0}.txt"), probabilities)
        np.savetxt(os.path.join(sn_folder_path, f"best_unitaries_{0}.txt"), best_unitaries)
        np.savetxt(os.path.join(sn_folder_path, f"state_parameters_point_{0}.txt"), np.vstack(data_points))
        np.savetxt(os.path.join(sn_folder_path, f"gtc_mean_{0}.txt"), gtc_mean)
        np.savetxt(os.path.join(sn_folder_path, f"gtp_mean_{0}.txt"), gtp_mean)
        np.savetxt(os.path.join(sn_folder_path, f"gtc_std_{0}.txt"), gtc_std)
        np.savetxt(os.path.join(sn_folder_path, f"gtp_std_{0}.txt"), gtp_std)
        with open(os.path.join(sn_folder_path, f'opt_data_{0}.pkl'), 'wb') as fp:
            pickle.dump(hyperparams, fp)

            # plot(sn_folder_path)

    return


def main():
    projectors = [np.array([1, 0, 1, 1]), np.zeros(4), np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1]),
                  np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 1]), np.array([0, 1, 1, 0]),
                  np.array([1, 1, 1, 0])]

    for i_p, p in enumerate(projectors):
        for gn in ["SU4", "circuit"]:
            run_simulation(state_names=["2-qubit-dm"],
                           n_points=1000,
                           n_iter=1,
                           n_guesses=1,
                           gate_name=gn,
                           seed=0,
                           proj_vec=p,
                           n_opt=10)


if __name__ == "__main__":
    main()
