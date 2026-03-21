# TODO: improved handling of complex valued data?

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import pickle
import numpy as np
import matplotlib.pyplot as plt

TRAINING_DATA = "training.npz"
PARAMS_DIR = "./"

# seems good for linear regression
# LEARNING_RATE = 0.5
# NUM_STEPS = 1000
# TRAINING_SET_PERCENTAGE = .6

LEARNING_RATE = 1e-2
NUM_STEPS = 500
TRAINING_SET_PERCENTAGE = .8


# TODO: no global
rng = jax.random.PRNGKey(0)
params_rng, split_rng = jax.random.split(rng, num=2)

def split_data(features, targets):
    data_size = targets.shape[0]
    n_training_samples = int(data_size * TRAINING_SET_PERCENTAGE)

    permutation = jax.random.permutation(split_rng, data_size)
    training_idxs = permutation[:n_training_samples]
    validation_idxs = permutation[n_training_samples:]
    
    data = {
        "train" : [features[training_idxs], targets[training_idxs]],
        "validation" : [features[validation_idxs], targets[validation_idxs]],
    }
    
    return data

def get_data(concatenate = False):
    data = np.load(TRAINING_DATA)

    def assert_real(signal):
        max_imag = np.abs(signal.imag).max()
        assert max_imag < 1e-10, f"Large imaginary parts in Chebyshev moments {max_imag}!"
        
    features = data["features"]
    assert_real(features)    
    if concatenate == True:
        n_samples, input_dim, _ = features.shape
        features = features.reshape(n_samples, 2 * input_dim)        

    # check if moments are sufficiently real
    targets = data["moments"]
    assert_real(targets)    

    return split_data(features.real, targets.real)

def plot_loss(name):
    loss = load_loss(name)
    plt.plot(np.arange(loss.size), loss)
    plt.title(f'Train loss for {name}')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.savefig(f"loss_{name}.pdf")
    plt.close()

def load_model(name):
    with open(f"{PARAMS_DIR}params_{name}.pkl", "rb") as f:
        params = pickle.load(f)
    return params

def save_model(params, name):
    with open(f"{PARAMS_DIR}params_{name}.pkl", "wb") as f:
        pickle.dump(params, f)

def load_loss(name):
    return jnp.load(f"{PARAMS_DIR}loss_{name}.npz")["loss"]

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

class MLP(nn.Module):
    n1 : int
    n2 : int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.n1, name = "dense1")(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=self.n2, name = "dense2")(x)
        x = nn.sigmoid(x)
        return x
    
def make_mlp():
    data = get_data(concatenate = True)
    features, targets = data["train"]
    
    # predict array of shape n_samples x n_batch
    n_out = targets.shape[-1]
    n_in = features.shape[-1]
    n_hidden = n_out + n_in #n_out #int((n_out + n_in) * 2/3)

    model = MLP(n1 = n_hidden, n2 = n_out)

    # parameters
    params = model.init(params_rng, jnp.ones((n_in,), dtype=jnp.float32))
    
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
            # return jnp.sum( (y-pred)**2 )
            return jnp.inner(y-pred, y-pred) / 2.0
        
        # Vectorizes the squared error and computes mean over the loss values.
        return jax.vmap(squared_error)(features, targets)
    
    if mean == True:
        # return jax.jit(lambda p : jnp.log(jnp.mean(sq_vec(p), axis = 0)) )
        return jax.jit(lambda p : jnp.mean(sq_vec(p), axis = 0))
    
    return jax.jit(sq_vec)


def run_linear_regression():
    model_name = "linear_regression"
    
    model, params = make_linear_regression()
    data = get_data(concatenate = True)
    features, targets = data["train"]

    run_training_loop(model, params, features, targets, model_name)    
    plot_loss(model_name)

    features, targets = data["validation"]
    validate(model, features, targets, model_name)    

def run_mlp():
    model_name = "mlp"
    
    model, params = make_mlp()
    data = get_data(concatenate = True)
    features, targets = data["train"]

    run_training_loop(model, params, features, targets, model_name)    
    plot_loss(model_name)

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
    loss_vals = loss(params)
    plt.xlabel("structure size")
    plt.ylabel("Validation loss")
    print(jnp.mean(loss_vals))
    plt.plot(np.arange(loss_vals.size), loss_vals)
    plt.savefig(f"loss_validation_{model_name}.pdf")
    plt.close()

if __name__ == '__main__':
    # run_linear_regression()
    run_mlp()
