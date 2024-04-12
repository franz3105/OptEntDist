import numpy as np
import os
import json
import jax
import datetime
import pickle
import h5py

from qutip_wrapper import werner, rho_zsolt, transformed_werner, tranformed_mau_state
from optimize_avg_c_protocol import werner_sample, mau_state_sample, xy_state_sample, xy_state_rot, optimize_protocol, \
    plot, concurrence, test_optimized_protocol
from load_2_qubit_dm import load_data


def run_simulation(state_names=None,
                   n_points=1,
                   n_iter=1,
                   n_guesses=1,
                   gate_name="SU4",
                   seed=0,
                   proj_vec=np.zeros(4),
                   n_opt=10, meas_p="argmax"):

    """

    Run the simulation for the optimization of the protocol.
    :param state_names: List of state names.
    :param n_points: Number of points in the data set.
    :param n_iter: Number of iterations.
    :param n_guesses: Number of guesses.
    :param gate_name: Name of the gate.
    :param seed: Seed for the random number generator.
    :param proj_vec: Projection vector.
    :param n_opt: Number of optimization steps.
    :param meas_p: Measurement protocol.
    :return:

    """
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
            dm_start, points = load_data("random_dm_2_new.csv", "random_a_vectors_new.csv")
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
                           n_components=n_components, gate_name=gate_name, seed=seed, proj_vector=proj_vec, n_opt=n_opt,
                           meas_p=meas_p)

        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sn_folder_path = os.path.join(data_folder, f"{sn}" + "_" + time_str)
        if not os.path.exists(sn_folder_path):
            os.mkdir(sn_folder_path)
        else:
            pass

        if not isinstance(dm_start, list):
            dm_start = [dm_start, ]
            points = [points, ]

        best_concurrences, probabilities, gate_params, data_points, gtc_mean, gtp_mean, gtc_var, gtp_var, all_c = \
            optimize_protocol(dm_start, points, n_iter, n_guesses, n_components, gate_name=gate_name,
                              proj_vec=proj_vec, n_idx=n_opt, meas_p=meas_p)

        # np.savetxt(os.path.join(sn_folder_path, "plotlines.txt"), plot_lines)
        np.savetxt(os.path.join(sn_folder_path, f"best_concurrences_{0}.txt"), best_concurrences)
        np.savetxt(os.path.join(sn_folder_path, f"avg_probabilities_{0}.txt"), probabilities)
        np.savetxt(os.path.join(sn_folder_path, f"best_params_{0}.txt"), gate_params)
        np.savetxt(os.path.join(sn_folder_path, f"state_parameters_point_{0}.txt"), np.vstack(data_points))
        np.savetxt(os.path.join(sn_folder_path, f"gtc_mean_{0}.txt"), gtc_mean)
        np.savetxt(os.path.join(sn_folder_path, f"gtp_mean_{0}.txt"), gtp_mean)
        np.savetxt(os.path.join(sn_folder_path, f"gtc_var_{0}.txt"), gtc_var)
        np.savetxt(os.path.join(sn_folder_path, f"gtp_var_{0}.txt"), gtp_var)
        np.savetxt(os.path.join(sn_folder_path, f"proj_vec{0}.txt"), proj_vec)

        # Save all_c as hdf5 file
        with h5py.File(os.path.join(sn_folder_path, f"all_c_{0}.hdf5"), 'w') as hf:
            hf.create_dataset("all_c", data=all_c)

        with open(os.path.join(sn_folder_path, f'opt_data_{0}.pkl'), 'wb') as fp:
            pickle.dump(hyperparams, fp)

        n_dm_files = 10  # Number of different data sets.
        n_points = int(1e6)  # Number of states in each data set.
        dm_start = []
        points = []
        fld = os.path.join(cwd, "data_all_dms")
        n_comp_1 = int(n_components / 2)

        for i in range(1, 1 + n_dm_files):
            # Tests average on 10 different data sets produced by the HR algorithm.

            dm_array, points_array = load_data(os.path.join(fld, f"random_dm_2_{i}_{n_points}.csv"),
                                               os.path.join(fld, f"random_a_vectors_{i}_{n_points}.csv"),
                                               n_points, a_size=30)
            gtc_mean_test, gtp_mean_test, gtc_var_test, gtp_var_test, all_c_test = \
                test_optimized_protocol(dm_array, n_iter, n_points, proj_vec, gate_params, gate_name, n_comp_1,
                                        meas_p=meas_p)

            np.savetxt(os.path.join(sn_folder_path, f"gtc_mean_test{i}.txt"), gtc_mean_test)
            np.savetxt(os.path.join(sn_folder_path, f"gtp_mean_test{i}.txt"), gtp_mean_test)
            np.savetxt(os.path.join(sn_folder_path, f"gtc_var_test{i}.txt"), gtc_var_test)
            np.savetxt(os.path.join(sn_folder_path, f"gtp_var_test{i}.txt"), gtp_var_test)

            # Save all_c_test as hdf5 file
            with h5py.File(os.path.join(sn_folder_path, f"all_c_test{i}.hdf5"), 'w') as hf:
                hf.create_dataset("all_c_test", data=all_c_test)

            dm_start.append(dm_array)
            points.append(points_array)
            # plot(sn_folder_path)

    return


def main():
    projectors = [np.array([1, 0, 1, 1]), ]  # , np.array([1, 1, 1, 1]),
    # np.array([1, 1, 0, 1]),
    # np.array([1, 1, 1, 0]), np.array([0, 1, 1, 1]),
    # np.array([1,0, 0, 1 ]), np.array([1, 0, 1, 0]),
    # np.array([1,1, 0, 0]),
    # np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])]

    meas_p = "zero"

    for i_p, p in enumerate(projectors):
        for gn in ["SU4", ]:
            for seed in range(5):
                run_simulation(state_names=["2-qubit-dm"],
                               n_points=1000,
                               n_iter=16,
                               n_guesses=5,
                               gate_name=gn,
                               seed=seed,
                               proj_vec=p,
                               n_opt=100,
                               meas_p=meas_p)


if __name__ == "__main__":
    main()
