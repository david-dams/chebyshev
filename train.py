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

MAX_FEATURES = 100

# TODO: no global
rng = jax.random.PRNGKey(0)
params_rng, split_rng = jax.random.split(rng, num=2)

## MODELS ##
class MLP(nn.Module):
    n1 : int
    n2 : int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.n1, name = "dense1")(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.n2, name = "dense2")(x)
        return x

class Conv(nn.Module):
    n_out : int

    @nn.compact
    def __call__(self, x):
        # dummy axes
        x = x[None, ..., None]

        # convolutional
        # N_C => 16 with kernel size K => 16 * K params
        x = nn.Conv(features=16, kernel_size=(3,), padding='SAME', name='conv1')(x)
        x = nn.gelu(x)
        x = nn.avg_pool(x, window_shape=(2,), strides = (2,), padding='SAME')
        x = nn.Conv(features=32, kernel_size=(3,), padding='SAME', name='conv2')(x)
        x = nn.gelu(x)
        x = nn.avg_pool(x, window_shape=(2,), strides = (2,), padding='SAME')

        # final shape: (N_C / 2**(N_pool), number_of_features_N) => global average over reduced feature axis
        x = x.mean(axis = 1)

        # small mlp head
        x = nn.Dense(features=self.n_out, name='dense1')(x)

        return x[0]
    
## DATA PREPARATION ##
def normalize(x, xm, xd):
    return (x-xm)/(xd + 1e-8)

def denormalize(x, xm, xd):
    return x*(xd + 1e-8) + xm

def split_data(features, targets, radii, corners):
    """split data randomly into training and validation set, returning dictionary with keys: train, validation and stats (mean, sd) of features and targets in training set to be used for inference"""
    features = jnp.array(features)
    targets = jnp.array(targets)
    radii = jnp.array(radii)
    corners = jnp.array(corners)
    
    data_size = targets.shape[0]
    n_training_samples = int(data_size * TRAINING_SET_PERCENTAGE)

    permutation = jax.random.permutation(split_rng, data_size)
    training_idxs = permutation[:n_training_samples]
    validation_idxs = permutation[n_training_samples:]

    ft, tt = features[training_idxs], targets[training_idxs]
    fv, tv = features[validation_idxs], targets[validation_idxs]
    targets_mean, targets_sd = jnp.mean(tt, axis = 0), jnp.std(tt, axis = 0)
    features_mean, features_sd = jnp.mean(ft, axis = 0), jnp.std(ft, axis = 0)    

    data = {
        "train" : [normalize(ft, features_mean, features_sd), normalize(tt, targets_mean, targets_sd)],
        "validation" : [normalize(fv, features_mean, features_sd), normalize(tv, targets_mean, targets_sd)],
        "radii_validation" : radii[validation_idxs],
        "corners_validation" : corners[validation_idxs],
        "radii_training" : radii[training_idxs],
        "corners_training" : corners[training_idxs],
        "targets_stats" : [targets_mean, targets_sd],
        "features_stats" : [features_mean, features_sd],
    }
    
    return data

def get_data():
    data = np.load(TRAINING_DATA)

    def assert_real(signal):
        max_imag = np.abs(signal.imag).max()
        assert max_imag < 1e-10, f"Large imaginary parts in Chebyshev moments {max_imag}!"
        
    features = data["features"]
    assert_real(features)    

    # check if moments are sufficiently real
    targets = data["moments"]
    assert_real(targets)

    return split_data(features.real, targets.real, data["radii"], data["corners"])

## VISUALIZATION ##
def plot_loss(name):
    loss = load_loss(name)
    plt.plot(np.arange(loss.size), loss)
    plt.title(f'Train loss for {name}')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.savefig(f"loss_{name}.pdf")
    plt.close()

## MODEL HANDLING ##
def sum_params_flat(x):
    """computes number of parameters in model"""
    if isinstance(x, dict):
        return sum(map(sum_params_flat, x.values()))
    else:
        return x.size
    
def load_model(name):
    with open(f"{PARAMS_DIR}params_{name}.pkl", "rb") as f:
        params = pickle.load(f)
    print(f"loading {name}, # params: {sum_params_flat(params)}")
    return params

def save_model(params, name):
    with open(f"{PARAMS_DIR}params_{name}.pkl", "wb") as f:
        pickle.dump(params, f)

def load_loss(name):
    return jnp.load(f"{PARAMS_DIR}loss_{name}.npz")["loss"]

def save_loss(loss, name):
    jnp.savez(f"{PARAMS_DIR}loss_{name}.npz", loss=jnp.array(loss))

## MODEL SETUP ##
def make_linear_regression():
    """linear regression computing

    c = M * x + b

    where c are chebyshev coefficients, M, b are learnable weights and biases and x represents shape via fourier descriptor (absolute values, shape is centered to ensure rotational invariance)   
    """
    data = get_data()
    features, targets = data["train"]
    
    # predict array of shape n_samples x n_batch
    model = nn.Dense(features=targets.shape[-1])

    # parameters
    params = model.init(params_rng, jnp.ones((features.shape[-1],), dtype=jnp.float32))
    
    return model, params
    
def make_mlp():
    data = get_data()
    features, targets = data["train"]
    
    # predict array of shape n_samples x n_batch
    n_out = targets.shape[-1]
    n_in = features.shape[-1]
    n_hidden = n_out + n_in #n_out #int((n_out + n_in) * 2/3)

    model = MLP(n1 = n_hidden, n2 = n_out)

    # parameters
    params = model.init(params_rng, jnp.ones((n_in,), dtype=jnp.float32))
    
    return model, params

def make_cnn():
    data = get_data()
    
    features, targets = data["train"]
    n_out = targets.shape[-1]
    n_in = features.shape[1]
    
    model = Conv(n_out = n_out)    

    # parameters
    params = model.init(params_rng, jnp.ones((n_in,), dtype=jnp.float32))
    
    return model, params

MODELS = {"cnn" : make_cnn, "mlp" : make_mlp, "linear" : make_linear_regression}

## LOSS FUNCTION ##
def make_cost_func(model, features, targets, mean = True):
    """cost function (mean squared error)

    model :
    features : 
    targets :

    returns JITed mse func
    """    
    def mse_vec(params):
        
        def mse(x, y):
            """mean squared error over model dimensions
            
            x: fourier shape descriptors of dim Nf
            y: chebyshev coefficients of dim Nt

            returns scalar: jnp.mean((y-y_pred)**2)
            """
            pred = model.apply(params, x)
            return jnp.mean((y-pred)**2)

        # vectorization: F, T -> E, where dims: N x Nf, N x Nt -> N
        return jax.vmap(mse)(features, targets)
    
    # for validation: compute per sample
    func = mse_vec
    
    # for training: average over samples to produce scalar
    if mean == True:
        func = lambda p : jnp.mean(mse_vec(p))

    return jax.jit(func)

## TRAIN + VALIDATE ##
def run_linear_regression():
    model_name = "linear_regression"
    
    model, params = make_linear_regression()
    data = get_data()
    features, targets = data["train"]

    run_training_loop(model, params, features, targets, model_name)    
    plot_loss(model_name)

    validate(model, data, model_name)    

def run_mlp():
    model_name = "mlp"
    
    model, params = make_mlp()
    data = get_data()
    features, targets = data["train"]

    run_training_loop(model, params, features, targets, model_name)    
    plot_loss(model_name)

    validate(model, data, model_name)    

def run_cnn():
    model_name = "cnn"
    
    model, params = make_cnn()
    data = get_data()
    features, targets = data["train"]

    # TODO: for whatever reason, JIT in scan is slower than plain Python (at least on CPU)
    run_training_loop(model, params, features, targets, model_name, use_scan = False)    
    plot_loss(model_name)

    validate(model, data, model_name)        


## GENERIC TRAINING ##
def get_training_func(tx, loss_grad_fn):
    
    # scan JITs => speeds up training
    def training_func(carry, loss):
        params, opt_state = carry
        loss_val, grads = loss_grad_fn(params)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    return training_func

def run_training_loop(model, params, features, targets, model_name, use_scan = True):
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

    if use_scan == True:
        f = get_training_func(tx, loss_grad_fn)
        carry, loss_history = jax.lax.scan(f,
                                           (params, opt_state),
                                           length = NUM_STEPS)
        params, _ = carry

    else:
        loss_history = []
        for _ in range(NUM_STEPS):
            loss_val, grads = loss_grad_fn(params)
            loss_history.append(loss_val)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        
    save_model(params, model_name)
    save_loss(loss_history, model_name)
    
## GENERIC VALIDATION ##
def r_squared(y, y_pred):
    """R^2 per chebyshev coefficient"""
    ss_res =  ((y-y_pred)**2).mean(axis = 0)
    ss_tot = y.var(axis = 0) + 1e-12
    return 1 - ss_res / ss_tot

def validate(model, data, model_name):
    x_std, y_std = data["validation"]
    x_mean, x_sigma = data["features_stats"]
    x = denormalize(x_std, x_mean, x_sigma)
    
    # standardized error
    params = load_model(model_name)        
    loss = make_cost_func(model, x_std, y_std, mean = False)
    loss_vals = loss(params)    
    plt.xlabel("Structure size")
    plt.ylabel("Validation loss (std)")
    print(model_name, ", validation loss (std): ", jnp.mean(loss_vals))
    plt.plot(x.sum(axis = 1), loss_vals)
    plt.savefig(f"std_loss_validation_{model_name}.pdf")
    plt.close()

    # unstandardized error
    y_mean, y_sigma = data["targets_stats"]
    y_pred = jax.vmap(lambda x : model.apply(params, x))(x_std)
    
    # denormalize
    y_pred = denormalize(y_pred, y_mean, y_sigma)
    y = denormalize(y_std, y_mean, y_sigma)

    # average over chebyshev coefficients
    loss_vals = jnp.mean((y - y_pred)**2, axis = 1)

    plt.xlabel("Structure size")
    plt.ylabel("Validation loss")    
    print(model_name, ", validation loss: ", jnp.mean(loss_vals))
    print(model_name, ", median R^2: ", jnp.median(r_squared(y, y_pred)))
    idxs = y_sigma.argsort()
    print(r_squared(y, y_pred)[idxs])
    plt.plot(x.sum(axis = 1), loss_vals)
    plt.savefig(f"loss_validation_{model_name}.pdf")
    plt.close()

    save_predictions(model_name, data)

def save_predictions(name, data):
    model, _ = MODELS[name]()
    params = load_model(name)

    x_std, y_std = data["validation"]
    y_mean, y_sigma = data["targets_stats"]
    y_pred = jax.vmap(lambda x : model.apply(params, x))(x_std)
    y_pred = denormalize(y_pred, y_mean, y_sigma)
    y = denormalize(y_std, y_mean, y_sigma)

    np.savez_compressed(
        f"{name}_predictions.npz",
        y = y,
        y_pred = y_pred,
        radii = data["radii_validation"],
        corners = data["corners_validation"]
    )
    
if __name__ == '__main__':
    # # simple baseline to test against
    run_linear_regression()

    # # small fcn
    run_mlp()

    # # exploit context
    run_cnn()
