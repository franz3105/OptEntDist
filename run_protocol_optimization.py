import numpy as np
import os
import json
import jax

from qutip_wrapper import werner, rho_zsolt, transformed_werner, tranformed_mau_state
from optimize_avg_c_protocol import werner_sample, mau_state_sample, xy_state_sample, xy_state_rot, optimize_protocol, \
    plot


def main():
    jax.config.update('jax_platforms', 'cpu')
    jax.config.update("jax_debug_nans", True)

    cwd = os.getcwd()
    data_folder = os.path.join(cwd, "data")
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    else:
        pass

    np.random.seed(0)
    # List of names of avaialble states
    # state_names = ["Werner"]
    state_names = ["XY_avg"]
    n_points = 5000
    n_iter = 5
    n_guesses = 5
    n_components = 15

    for sn in state_names:
        if sn == "Werner":
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
                           n_components=n_components)

        sn_folder_path = os.path.join(data_folder, f"{sn}")
        if not os.path.exists(sn_folder_path):
            os.mkdir(sn_folder_path)
        else:
            pass

        for _ in range(10):
            plot_lines, best_concurrences, probabilities, best_unitaries, data_points = \
                optimize_protocol(dm_start, points, n_iter, n_guesses, n_components)
            print(best_concurrences)

            # np.savetxt(os.path.join(sn_folder_path, "plotlines.txt"), plot_lines)
            np.savetxt(os.path.join(sn_folder_path, "best_concurrences.txt"), best_concurrences)
            np.savetxt(os.path.join(sn_folder_path, "avg_probabilities.txt"), probabilities)
            np.savetxt(os.path.join(sn_folder_path, "best_unitaries.txt"), best_unitaries)
            np.savetxt(os.path.join(sn_folder_path, "state_parameters_point.txt"), np.vstack(data_points))
            with open(os.path.join(sn_folder_path, 'opt_data.json'), 'w') as fp:
                json.dump(hyperparams, fp)

            #plot(sn_folder_path)

    return


if __name__ == "__main__":
    main()
