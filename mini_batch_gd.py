import numpy as np
from jax import grad, value_and_grad
from jax.experimental import optimizers


# function to calculate cost
def create_mini_batches(data, batch_size=32):
    mini_batches = []
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, ::]
        #print(mini_batch.shape)
        mini_batches.append(mini_batch)

    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        mini_batches.append(mini_batch)

    return mini_batches


# function to perform mini-batch gradient descent
class Output:
    def __init__(self, fun, x):
        self.fun = fun
        self.x = x

def gradient_descent(cost, data_points, learning_rate=0.001, batch_size=32):
    theta = np.zeros((data_points.shape[1], 1))
    error_list = []
    max_iters = 3
    cost_grad = grad(cost)
    for itr in range(max_iters):
        mini_batches = create_mini_batches(data_points, batch_size=batch_size)
        for mini_batch in mini_batches:
            theta = theta - learning_rate * cost_grad(mini_batch, theta)
            error_list.append(cost(mini_batch, theta))

    return theta, error_list


def minimize_adam(cost, params, data_points, learning_rate=1e-3, batch_size=100):

    error_list = []
    max_iters = 10000
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(params)

    def cost_function(par):
        return cost(par, data_points)

    for itr in range(max_iters):
        mini_batches = create_mini_batches(data_points, batch_size=batch_size)
        for mini_batch in mini_batches:
            value, grads = value_and_grad(lambda x: cost(x, mini_batch))(get_params(opt_state))
            print(value)
            opt_state = opt_update(itr, grads, opt_state)
            # theta = theta - learning_rate * cost_grad(mini_batch, theta)
            if itr % 10 == 0:
                print(f"Cost function:", cost_function(get_params(opt_state)))

            error_list.append(cost(get_params(opt_state), data_points))

    params = get_params(opt_state)

    return Output(cost_function, params), error_list
