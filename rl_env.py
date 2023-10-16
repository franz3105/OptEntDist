import gymnasium
import numpy as np
import qiskit as qk
import jax
import jax.numpy as jnp

from jax import jit
from functools import partial
from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Operator
from gymnasium import spaces
from optimize_avg_c_protocol import sampled_av_gate_inconcurrence, concurrence, partial_transpose, purification, CNOT, \
    X, Y, Z
from load_2_qubit_dm import load_data


def layer_circuit(n_qubits=2, n_layers=1, param_name='theta', analog_tqg=False):
    qc = QuantumCircuit(n_qubits)

    if analog_tqg:
        N = int(n_qubits * (n_qubits - 1) / 2 + 2 * n_qubits) * n_layers
        # N = 2*n_qubits*n_layers
    else:
        N = 4 * n_qubits * n_layers

    theta_param = ParameterVector(param_name, length=N)
    counter = 0

    if analog_tqg:
        for k in range(n_layers):
            for iq in range(n_qubits):
                qc.rx(theta_param[counter], iq)
                qc.rz(theta_param[counter + 1], iq)
                counter += 2
                for jq in range(iq):
                    qc.rxx(theta_param[counter], qubit1=iq, qubit2=jq)
                    # qc.ryy(theta_param[counter+1], qubit1=iq, qubit2=jq)
                    counter += 1

    else:
        for k in range(n_layers):
            for iq in range(n_qubits):
                qc.rx(theta_param[counter], iq)
                qc.rz(theta_param[counter + 1], iq)
                counter += 2
                for jq in range(iq):
                    qc.cnot(jq, iq)
                qc.rx(theta_param[counter], iq)
                qc.rz(theta_param[counter + 1], iq)
                counter += 2
                for jq in range(iq):
                    qc.cnot(iq, jq)

    # for iq in range(n_qubits):
    #    qc.ry(theta_param[counter], iq)
    #    qc.rx(theta_param[counter + 1], iq)
    #    counter += 2

    ##assert counter == N
    # print(len(theta_param))
    # print(counter)

    return qc, theta_param, N


def layered_circuit_unitary(n_qubits, n_layers, theta):
    qc, theta_param, N = layer_circuit(n_qubits=n_qubits, n_layers=n_layers, param_name='theta')
    qc.assign_parameters({theta_param: theta})

    return UnitaryGate(qc)


class CustomEnv(gymnasium.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_qubits, len_circuit, n_gates, dm_dataset):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        np.random.seed(0)
        # Example when using discrete actions:
        self.action_space = spaces.MultiDiscrete(np.array([n_gates, n_qubits, n_qubits, 2]))
        # Example for using image as input:
        self.n_qubits = n_qubits
        self.len_circuit = len_circuit
        self.n_gates = n_gates
        self.max_actions = 2*len_circuit

        self.observation_space = spaces.Dict({'value': spaces.Box(low=0, high=255,
                                                                  shape=(n_qubits, len_circuit, n_gates, 2),
                                                                  dtype=np.uint8)})

        self.matrix_circuit = np.zeros((n_qubits, len_circuit, n_gates, 2), dtype=np.uint8)
        self.qc_circuit = qk.QuantumCircuit(n_qubits)
        # c_dm = jax.vmap(concurrence)(dm_dataset)
        # idx_non_zero = np.where(c_dm > 1e-15)[0]
        self.dm_data_set = dm_dataset
        # points = points[idx_non_zero]
        self.dataset_size = self.dm_data_set.shape[0]
        self.Id4 = np.identity(4)
        self.fom_data = []
        self.circuits = []
        self.n_interactions = 0
        self.n_points = 100
        self.idx_random = np.random.choice(self.dm_data_set.shape[0], self.n_points, replace=False)
        self.rho_test = self.dm_data_set[self.idx_random, ::]

    def sample_dm(self):
        idx = np.random.randint(0, self.dataset_size, size=1)
        return self.dm_data_set[idx, ::]

    def concurrence_diff(self, qc_A, qc_B):
        dm = self.sample_dm()
        c_start = 1 - sampled_av_gate_inconcurrence(dm, self.Id4, self.Id4)
        # print(f"Concurrence: ", c_start)
        #qc_A.cnot(1, 0)
        #qc_B.cnot(1, 0)

        U_A = Operator(qc_A).data
        # print(U_A)
        U_B = Operator(qc_B).data
        # print(U_B)
        c_end = 1 - sampled_av_gate_inconcurrence(dm, U_A, U_B)
        # print(c_end)
        return c_end - c_start

    def reward_concurrence_diff(self, qc_A, qc_B):
        delta_c = float(self.concurrence_diff(qc_A, qc_B))
        reward = delta_c
        return reward

    def entangled_purity(self, qc_A, qc_B):

        dm = self.sample_dm()[0]
        purity_start = np.real(np.trace(dm.dot(dm)))

        U_A = Operator(qc_A).data
        U_B = Operator(qc_B).data

        dm_after = purification(dm, U_A, U_B)[2]

        eigs = np.linalg.eigvalsh(partial_transpose(dm_after))

        if np.all(np.greater_equal(eigs, 0)):
            reward = np.real(np.trace(dm_after.dot(dm_after)))
        else:
            reward = -1

        return reward

    def save_fom_data(self, reward, c_A, c_B):
        self.fom_data.append(reward)
        self.circuits.append([c_A, c_B])

    @partial(jit, static_argnums=(0,))
    def average_concurrence(self, dm_sample):

        c_avg = jax.vmap(concurrence)(dm_sample)

        return jnp.mean(c_avg)

    @partial(jit, static_argnums=(0,))
    def purify_ensemble(self, U_A, U_B):
        return jax.vmap(lambda x: purification(x, U_A, U_B)[2])(self.rho_test)

    def test_concurrence(self, qc_A, qc_B):

        U_A = Operator(qc_A).data
        U_B = Operator(qc_B).data
        dm_after = jax.vmap(lambda x: purification(x, U_A, U_B)[2])(self.rho_test)
        c_avg = self.average_concurrence(dm_after)
        return c_avg

    def step(self, action):
        """Take one action and return observation, reward, done, info"""
        self.matrix_circuit = self.from_action_to_matrix(action)
        qc_A, qc_B = self.from_matrix_to_recovery_circuit(build_qc=True)

        # delta_c = float(self.concurrence_diff(qc_A, qc_B))
        # dm = self.sample_dm()[0]
        reward = float(self.concurrence_diff(qc_A, qc_B))

        #self.save_fom_data(reward, qc_A, qc_B)
        self.n_interactions += 1
        #if self.n_interactions % 2000 == 0:
        #    c_avg = self.test_concurrence(qc_A, qc_B)
        #    print(c_avg)

        # print(reward)
        observation = {'value': self.matrix_circuit}
        done = False

        if self.n_interactions % self.max_actions == 0:
            done = True

        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def reset(self, seed=0):
        #np.random.seed(seed)
        #self.n_interactions = 0
        self.matrix_circuit = np.zeros((self.n_qubits, self.len_circuit, self.n_gates, 2), dtype=np.uint8)
        observation = {'value': self.matrix_circuit}
        info = {}

        return observation, info  # reward, done, info can't be included

    def render(self, mode='human'):
        return

    def close(self):
        return

    def from_action_to_matrix(self, action):
        """
        An action wil be in the following form [port, spot, t, c]
        we recall that in every spot can stand just one port. In this function we request as input the matrix at time t-1
        and the action performed. We will output the matrix at time t.
        The index of the ports are:
        0: X Pauli operation
        1: Z Pauli operation
        2: CNOT operation
        """
        spot = self.n_interactions % self.len_circuit
        port, target, control, subsystem = action.astype(int)[:]  # We impose to have only one gate to splot
        self.matrix_circuit[:, spot, :, subsystem] = 0  # Null action
        if port == 4:
            return self.matrix_circuit  # We put the gate in the circuit
        self.matrix_circuit[target, spot, port, subsystem] = 1
        # Check if CNOT has same target and control qubit
        if port == 2 and control != target:
            self.matrix_circuit[control, spot, port, subsystem] = 2
        elif port == 2 and control == target:
            self.matrix_circuit[target, spot, port, subsystem] = 0
        return self.matrix_circuit

    def from_matrix_to_recovery_circuit(self, build_qc=True):
        """
        In this function we will take as input a n_qubits x spot x operation where:
            n_qubits : number of qubits
            spot : position in the final circuit
            operation : possible operation
        and we will return a quantum circuit. In particular the operations are encoded in the following way:
            0: X Pauli operation
            1: Z Pauli operation
            2: Y Pauli operation
            3: CNOT operation
        """  # From the matrix we extrapolate the dimensions

        if build_qc:
            # We create an empty quantum circuit
            qubits, spots, operations = np.shape(self.matrix_circuit[::, 0])
            codeword_qubit_A = QuantumRegister(qubits, 'code_qubits')
            qc_A = QuantumCircuit(codeword_qubit_A)  # Check the trivial condition: The matrix is made of all zeros

            qubits, spots, operations = np.shape(self.matrix_circuit[::, 1])
            codeword_qubit_B = QuantumRegister(qubits, 'code_qubits')
            qc_B = QuantumCircuit(codeword_qubit_B)  # Check the trivial condition: The matrix is made of all zeros
            qc_tuple = (qc_A, qc_B)
            cdw_tuple = (codeword_qubit_A, codeword_qubit_B)

            if not np.any(self.matrix_circuit):
                return qc_A, qc_B

            for i_qc, qc in enumerate(qc_tuple):
                for spot in range(spots):
                    for qubit in range(qubits):
                        for operation in range(operations):
                            # print(qubit, spot, operation, i_qc)
                            if self.matrix_circuit[qubit, spot, operation, i_qc] == 1:
                                # Check for Hadamart
                                if operation == 0:
                                    qc.h(cdw_tuple[i_qc][qubit])
                                # Check for S gate
                                elif operation == 1:
                                    qc.s(cdw_tuple[i_qc][qubit])
                                # Check CNOT gate
                                elif operation == 2:
                                    t_qubit = qubit
                                    c_qubit = self.control_qubit(spot, operation)
                                    qc.cnot(cdw_tuple[i_qc][c_qubit], cdw_tuple[i_qc][t_qubit])
                                # Check for T gate
                                elif operation == 3:
                                    qc.t(cdw_tuple[i_qc][qubit])
                    # qc.barrier()

            return qc_A, qc_B
        else:
            raise NotImplementedError("Unknown")

    def control_qubit(self, spot, operation):
        index = np.where(self.matrix_circuit[:, spot, operation] == 2)
        return index[0][0]


def run_rl_training():
    n_qubits = 2
    max_depth = 50
    n_gates = 4
    dm_start, points = load_data("random_dm_2.csv", "random_a_vectors.csv")

    env = CustomEnv(n_qubits=n_qubits, len_circuit=max_depth, n_gates=n_gates, dm_dataset=dm_start)
    env.reset(seed=0)
    env.step(np.array([0, 1, 1, 1], np.int8))

    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    # Instantiate the env
    # Define and Train the agent
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./ppo_ppt_purity_tensorboard/")

    # model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=1000000, tb_log_name="first_run")
    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    # By default, it will create a new curve
    # model.learn(total_timesteps=10000, tb_log_name="second_run", reset_num_timesteps=False)
    # model.learn(total_timesteps=10000, tb_log_name="thrid_run", reset_num_timesteps=False)


if __name__ == '__main__':
    run_rl_training()
