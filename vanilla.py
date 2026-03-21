# fits a vanilla model to the data

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from flax import linen as nn

import numpy as np
import matplotlib.pyplot as plt


LEARNING_RATE = 1e-2
NUM_STEPS = 400

rng = jax.random.PRNGKey(0)
params_rng, w_rng, b_rng, samples_rng, noise_rng = jax.random.split(rng, num=5)

def make_model(n_shape, n_moments):
    # Creates a one linear layer instance.
    model = nn.Dense(features=n_moments)

    # Initializes the parameters.
    params = model.init(params_rng, jnp.ones((n_shape,), dtype=jnp.float32))
    
    return model, params

def make_mse_func(model, x_batched, y_batched):
  def mse(params):
    # Defines the squared loss for a single (x, y) pair.
    def squared_error(x, y):
      pred = model.apply(params, x)
      return jnp.inner(y-pred, y-pred) / 2.0
    # Vectorizes the squared error and computes mean over the loss values.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)
  return jax.jit(mse)  # `jit`s the result.

training_name = "training.npz"
data = np.load(training_name)

model, params = make_model(data["features"].shape[-1] * 2, data["moments"].shape[-1])

# Instantiates the sampled loss.
n_samples = data["features"].shape[0]
dim = data["features"].shape[1] * data["features"].shape[2]
loss = make_mse_func(model, data["features"].reshape(n_samples, dim), data["moments"].real)

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

plt.plot(loss_history)
plt.title('Train loss')
plt.xlabel('Step')
plt.ylabel('MSE')
plt.savefig("loss.pdf")

