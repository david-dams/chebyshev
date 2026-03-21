# TODO: improved handling of complex valued data?

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from flax import linen as nn

import numpy as np
import matplotlib.pyplot as plt

TRAINING_DATA = "training.npz"
PARAMS_DIR = "./"
LEARNING_RATE = 1e-2
NUM_STEPS = 400
TRAINING_SET_PERCENTAGE = .9


# TODO: no global
rng = jax.random.PRNGKey(0)
params_rng, w_rng, b_rng, samples_rng, noise_rng = jax.random.split(rng, num=5)

def split_data(features, targets):
    data_size = targets.shape[0]
    n_training_samples = int(data_size * TRAINING_SET_PERCENTAGE)

    permutation = jax.random.permutation(data_size)
    training_idxs = permutation[:n_training_samples]
    validation_idxs = permutation[n_training_samples:]
    
    data = {
        "train" : [features[training_idxs], targets[training_idxs]],
        "validation" : [features[validation_idxs], targets[validation_idxs]],
    }
    
    return data

def get_data(concatenate = False):
    data = np.load(TRAINING_DATA)

    features = data["features"]
    if concatenate == True:
        n_samples, _, input_dim = data["features"].shape
        features = features.reshape(n_samples, 2 * input_dim)

    # check if moments are sufficiently real
    targets = data["moments"]
    max_imag = np.abs(targets.imag).max()
    assert max_imag < 1e-10, f"Large imaginary parts in Chebyshev moments {max_imag}!"

    return split_data(features, targets.real)

def plot_loss(name):
    loss = load_loss(name)    
    plt.plot(loss)
    plt.title(f'Train loss for {name}')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.savefig(f"loss_{name}.pdf")

def load_model(name):
    with open(f"{PARAMS_DIR}params_{name}.pkl", "rb") as f:
        params = pickle.load(f)
    return params

def save_model(params, name):
    with open(f"{PARAMS_DIR}params_{name}.pkl", "wb") as f:
        pickle.dump(params, f)

def load_loss(name):
    return jnp.load(f"{PARAMS_DIR}loss_{name}.npz")

def save_loss(loss, name):
    jnp.savez(f"{PARAMS_DIR}loss_{name}.npz", loss=jnp.array(loss))

def make_linear_regression():
    """linear regression computing

    c = M * x + b

    where c are chebyshev coefficients, M, b are learnable weights and biases and x is the concatenated input vector of radius and phase
    
    x = [radius, phase]
    """
    data = get_data(concatenate = True)
    features, targets = data["train"]
    
    # predict array of shape n_samples x n_batch
    model = nn.Dense(features=targets.shape[-1])

    # parameters
    params = model.init(params_rng, jnp.ones((features.shape[-1],), dtype=jnp.float32))
    
    return model, params

def make_cost_func(model, features, targets, mean = True):
    """cost function (mean squared error)

    model :
    features : 
    targets :

    returns JITed mse func
    """    
    def sq_vec(params):
        
        # Defines the squared loss for a single (x, y) pair.
        def squared_error(x, y):
            pred = model.apply(params, x)
            return jnp.inner(y-pred, y-pred) / 2.0
        
        # Vectorizes the squared error and computes mean over the loss values.
        return jax.vmap(squared_error)(features, targets)
    
    if mean == True:
        return jnp.mean(jax.jit(sq_vec), axis = 0)
    
    return jax.jit(sq_vec)


def run_linear_regression():
    model_name = "linear_regression"
    
    model, params = make_linear_regression()
    data = get_data(concatenate = True)
    features, targets = data["train"]

    run_training_loop(model, params, features, targets, model_name)    
    plot_loss("linear_regression")

    features, targets = data["validation"]
    validate(model, features, targets, model_name)    

def run_training_loop(model, params, features, targets, model_name):
    """trains model and saves params
    """
    
    loss = make_cost_func(model, features, targets)

    # Creates a function that returns value and gradient of the loss.
    loss_grad_fn = jax.value_and_grad(loss)

    tx = optax.chain(
        # Sets the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        # Puts a minus sign to *minimize* the loss.
        optax.scale(-LEARNING_RATE)
    )

    opt_state = tx.init(params)

    loss_history = []

    # Minimizes the loss.
    for _ in range(NUM_STEPS):
      # Computes gradient of the loss.
      loss_val, grads = loss_grad_fn(params)
      loss_history.append(loss_val)
      # Updates the optimizer state, creates an update to the params.
      updates, opt_state = tx.update(grads, opt_state)
      # Updates the parameters.
      params = optax.apply_updates(params, updates)

    save_model(params, model_name)
    save_loss(loss_history, model_name)

def validate(model, features, targets, model_name):
    params = load_model(model_name)    
    loss = make_cost_func(model, features, targets, mean = False)
    
    plt.plot(loss(params))
    plt.savefig(f"loss_validation_{model_name}.pdf")
