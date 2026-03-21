import os
import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import optax
import matplotlib.pyplot as plt


TRAINING_DATA = "training.npz"
PARAMS_DIR = "params/"
PLOT_DIR = "plots/"
PREDICTIONS_DIR = "predictions/"

os.makedirs(PARAMS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

LEARNING_RATE = 1e-2
NUM_STEPS = 500
TRAINING_SET_PERCENTAGE = 0.8

HOLDOUT_RADIUS_RANGE = (20.0, 40.0)   
HOLDOUT_MODE = "radius_interval"  # "random"
BATCH_SIZE = 128
SHUFFLE_EACH_EPOCH = True

MAX_FEATURES = 100
N_MOMENTS = 100
NORMALIZE_AB = True

CNN = "cnn"
VANILLA = "mlp"
LINEAR_REGRESSION = "linear_regression"

rng = jax.random.PRNGKey(0)
params_rng, split_rng = jax.random.split(rng, 2)


# ----------------------------
# Models
# ----------------------------
class Linear(nn.Module):
    n_out: int

    @nn.compact
    def __call__(self, x):
        mu = nn.Dense(features=self.n_out, name="chebyshev")(x)
        ab = nn.Dense(features=2, name="energy")(x)
        return {"mu": mu, "ab": ab}


class MLP(nn.Module):
    n_hidden: int
    n_out: int

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(features=self.n_hidden, name="dense1")(x)
        h = nn.gelu(h)
        mu = nn.Dense(features=self.n_out, name="dense_chebyshev")(h)
        ab = nn.Dense(features=2, name="dense_energy")(h)
        return {"mu": mu, "ab": ab}


class Conv(nn.Module):
    n_out: int

    @nn.compact
    def __call__(self, x):
        # x: (n_features,)
        x = x[None, :, None]  # -> (1, length, channels)

        x = nn.Conv(features=16, kernel_size=(3,), padding="SAME", name="conv1")(x)
        x = nn.gelu(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,), padding="SAME")

        x = nn.Conv(features=32, kernel_size=(3,), padding="SAME", name="conv2")(x)
        x = nn.gelu(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,), padding="SAME")

        x = x.mean(axis=1)  # global average over length axis -> (1, channels)

        mu = nn.Dense(features=self.n_out, name="dense_chebyshev")(x)[0]
        ab = nn.Dense(features=2, name="dense_energy")(x)[0]
        return {"mu": mu, "ab": ab}


# ----------------------------
# Utilities
# ----------------------------
def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)


def denormalize(x, mean, std):
    return x * (std + 1e-8) + mean


def assert_real(name, arr, tol=1e-10):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        max_imag = np.abs(arr.imag).max()
        assert max_imag < tol, f"{name} has large imaginary part: {max_imag}"


def sum_params_flat(x):
    if isinstance(x, dict):
        return sum(map(sum_params_flat, x.values()))
    return x.size


def save_model(params, name):
    with open(os.path.join(PARAMS_DIR, f"params_{name}.pkl"), "wb") as f:
        pickle.dump(params, f)


def load_model(name):
    with open(os.path.join(PARAMS_DIR, f"params_{name}.pkl"), "rb") as f:
        params = pickle.load(f)
    print(f"loaded {name}, #params = {sum_params_flat(params)}")
    return params


def save_loss(loss_history, name):
    np.savez(os.path.join(PARAMS_DIR, f"loss_{name}.npz"), loss=np.asarray(loss_history))


def load_loss(name):
    return np.load(os.path.join(PARAMS_DIR, f"loss_{name}.npz"))["loss"]


def plot_loss(name):
    loss = load_loss(name)
    plt.plot(np.arange(loss.size), loss)
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.title(f"Training loss: {name}")
    plt.savefig(os.path.join(PLOT_DIR, f"loss_{name}.pdf"))
    plt.close()


# ----------------------------
# Data container
# ----------------------------
@dataclass
class Dataset:
    train: dict
    validation: dict
    x_mean: jnp.ndarray
    x_std: jnp.ndarray
    mu_mean: jnp.ndarray
    mu_std: jnp.ndarray
    ab_mean: jnp.ndarray
    ab_std: jnp.ndarray


# ----------------------------
# Data loading / splitting
# ----------------------------
def split_data(features, moments, radii, corners, lower_limits, upper_limits,
               holdout_mode="random",
               holdout_radius_range=None):
    x = jnp.asarray(features, dtype=jnp.float32)
    mu = jnp.asarray(moments, dtype=jnp.float32)
    radii = jnp.asarray(radii, dtype=jnp.float32)
    corners = jnp.asarray(corners, dtype=jnp.float32)

    ab = jnp.stack(
        [
            jnp.asarray(lower_limits, dtype=jnp.float32),
            jnp.asarray(upper_limits, dtype=jnp.float32),
        ],
        axis=1,
    )

    n_samples = x.shape[0]

    if holdout_mode == "random":
        n_train = int(TRAINING_SET_PERCENTAGE * n_samples)
        perm = jax.random.permutation(split_rng, n_samples)
        idx_train = perm[:n_train]
        idx_val = perm[n_train:]

    elif holdout_mode == "radius_interval":
        if holdout_radius_range is None:
            raise ValueError("holdout_radius_range must be provided for radius_interval mode")

        rmin, rmax = holdout_radius_range
        val_mask = (radii >= rmin) & (radii <= rmax)
        train_mask = ~val_mask

        idx_train = jnp.where(train_mask)[0]
        idx_val = jnp.where(val_mask)[0]

        if idx_train.size == 0:
            raise ValueError("No training samples left after radius holdout")
        if idx_val.size == 0:
            raise ValueError("No validation samples in requested radius holdout range")

    else:
        raise ValueError(f"Unknown holdout_mode: {holdout_mode}")

    def stats(arr, idx):
        return jnp.mean(arr[idx], axis=0), jnp.std(arr[idx], axis=0)

    x_mean, x_std = stats(x, idx_train)
    mu_mean, mu_std = stats(mu, idx_train)
    ab_mean, ab_std = stats(ab, idx_train)

    def pack(idxs):
        x_part = normalize(x[idxs], x_mean, x_std)
        mu_part = normalize(mu[idxs], mu_mean, mu_std)
        ab_part = normalize(ab[idxs], ab_mean, ab_std) if NORMALIZE_AB else ab[idxs]

        return {
            "fourier": x_part,
            "mu": mu_part,
            "ab": ab_part,
            "radii": radii[idxs],
            "corners": corners[idxs],
        }

    return Dataset(
        train=pack(idx_train),
        validation=pack(idx_val),
        x_mean=x_mean,
        x_std=x_std,
        mu_mean=mu_mean,
        mu_std=mu_std,
        ab_mean=ab_mean,
        ab_std=ab_std,
    )

def get_data():
    raw = np.load(TRAINING_DATA, allow_pickle=False)

    features = raw["features"][..., :MAX_FEATURES]
    moments = raw["moments"][..., :N_MOMENTS]

    assert_real("features", features)
    assert_real("moments", moments)

    features = np.asarray(features.real, dtype=np.float32)
    moments = np.asarray(moments.real, dtype=np.float32)

    radii = np.asarray(raw["radii"], dtype=np.float32)
    corners = np.asarray(raw["corners"], dtype=np.float32)

    if "lower_limits" in raw and "upper_limits" in raw:
        lower_limits = np.asarray(raw["lower_limits"], dtype=np.float32)
        upper_limits = np.asarray(raw["upper_limits"], dtype=np.float32)
    else:
        lower_limits = np.zeros(features.shape[0], dtype=np.float32)
        upper_limits = np.zeros(features.shape[0], dtype=np.float32)

    return split_data(
        features,
        moments,
        radii,
        corners,
        lower_limits,
        upper_limits,
        holdout_mode=HOLDOUT_MODE,
        holdout_radius_range=HOLDOUT_RADIUS_RANGE,
    )

def get_features_targets(split):
    x = {"fourier": split["fourier"]}
    y = {"mu": split["mu"], "ab": split["ab"]}
    return x, y


# ----------------------------
# Model construction
# ----------------------------
def make_model(model_name, data: Dataset):
    x_train, y_train = get_features_targets(data.train)
    n_in = x_train["fourier"].shape[-1]
    n_out = y_train["mu"].shape[-1]

    if model_name == LINEAR_REGRESSION:
        model = Linear(n_out=n_out)
    elif model_name == VANILLA:
        n_hidden = n_in + n_out
        model = MLP(n_hidden=n_hidden, n_out=n_out)
    elif model_name == CNN:
        model = Conv(n_out=n_out)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    params = model.init(params_rng, jnp.ones((n_in,), dtype=jnp.float32))
    return model, params


# ----------------------------
# Loss / training
# ----------------------------
def make_loss_fn(model, features, targets, ab_weight=1.0):
    def loss_fn(params):
        def per_sample(x, y):
            pred = model.apply(params, x["fourier"])
            err_mu = jnp.mean((y["mu"] - pred["mu"]) ** 2)
            err_ab = jnp.mean((y["ab"] - pred["ab"]) ** 2)
            return err_mu + ab_weight * err_ab

        return jnp.mean(jax.vmap(per_sample)(features, targets))

    return jax.jit(loss_fn)


def get_training_step(tx, loss_grad_fn):
    def step(carry, _):
        params, opt_state = carry
        loss_val, grads = loss_grad_fn(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    return step


def run_training_loop(model, params, data: Dataset, model_name, use_scan=True, ab_weight=1.0):
    x_train, y_train = get_features_targets(data.train)

    loss_fn = make_loss_fn(model, x_train, y_train, ab_weight=ab_weight)
    loss_grad_fn = jax.value_and_grad(loss_fn)

    tx = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = tx.init(params)

    if use_scan:
        step_fn = get_training_step(tx, loss_grad_fn)
        (params, _), loss_history = jax.lax.scan(
            step_fn,
            (params, opt_state),
            xs=None,
            length=NUM_STEPS,
        )
    else:
        loss_history = []
        for _ in range(NUM_STEPS):
            loss_val, grads = loss_grad_fn(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            loss_history.append(loss_val)
        loss_history = jnp.asarray(loss_history)

    save_model(params, model_name)
    save_loss(loss_history, model_name)
    return params


# ----------------------------
# Validation
# ----------------------------
def r_squared(y, y_pred):
    ss_res = ((y - y_pred) ** 2).mean(axis=0)
    ss_tot = y.var(axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def predict_split(model, params, split, data: Dataset, denorm_mu=True, denorm_ab=True):
    x, y = get_features_targets(split)
    pred = jax.vmap(lambda xi: model.apply(params, xi))(x["fourier"])

    y_mu = y["mu"]
    y_ab = y["ab"]
    y_pred_mu = pred["mu"]
    y_pred_ab = pred["ab"]

    if denorm_mu:
        y_mu = denormalize(y_mu, data.mu_mean, data.mu_std)
        y_pred_mu = denormalize(y_pred_mu, data.mu_mean, data.mu_std)

    if denorm_ab and NORMALIZE_AB:
        y_ab = denormalize(y_ab, data.ab_mean, data.ab_std)
        y_pred_ab = denormalize(y_pred_ab, data.ab_mean, data.ab_std)

    return {
        "y_mu": y_mu,
        "y_pred_mu": y_pred_mu,
        "y_ab": y_ab,
        "y_pred_ab": y_pred_ab,
        "radii": split["radii"],
        "corners": split["corners"],
    }


def compute_errors(pred_dict):
    err_mu_per_sample = jnp.mean((pred_dict["y_mu"] - pred_dict["y_pred_mu"]) ** 2, axis=1)
    err_ab_per_sample = jnp.mean((pred_dict["y_ab"] - pred_dict["y_pred_ab"]) ** 2, axis=1)
    r2_mu = r_squared(pred_dict["y_mu"], pred_dict["y_pred_mu"])
    return {
        "mu": err_mu_per_sample,
        "ab": err_ab_per_sample,
        "r2_mu": r2_mu,
    }


def plot_validation_vs_radius(radii, err, ylabel, filename):
    radii = np.asarray(radii)
    err = np.asarray(err)
    idx = np.argsort(radii)

    plt.scatter(radii[idx], err[idx], s=6)
    plt.xlabel("radius")
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


def validate(model, data: Dataset, model_name):
    params = load_model(model_name)

    pred = predict_split(model, params, data.validation, data, denorm_mu=False, denorm_ab=False)
    err_std_mu = jnp.mean((pred["y_mu"] - pred["y_pred_mu"]) ** 2, axis=1)
    plot_validation_vs_radius(
        data.validation["radii"],
        err_std_mu,
        ylabel="moment MSE (standardized)",
        filename=f"{PLOT_DIR}{model_name}_err_std.pdf",
    )

    pred = predict_split(model, params, data.validation, data, denorm_mu=True, denorm_ab=True)
    err = compute_errors(pred)

    print(model_name, "validation mean MSE(mu):", float(jnp.mean(err["mu"])))
    print(model_name, "validation mean MSE(ab):", float(jnp.mean(err["ab"])))
    print(model_name, "validation median R^2(mu):", float(jnp.median(err["r2_mu"])))

    plot_validation_vs_radius(
        pred["radii"],
        err["mu"],
        ylabel="moment MSE",
        filename=f"{PLOT_DIR}{model_name}_err.pdf",
    )

    np.savez_compressed(
        f"{PREDICTIONS_DIR}{model_name}.npz",
        y_mu=np.asarray(pred["y_mu"]),
        y_pred_mu=np.asarray(pred["y_pred_mu"]),
        y_ab=np.asarray(pred["y_ab"]),
        y_pred_ab=np.asarray(pred["y_pred_ab"]),
        radii=np.asarray(pred["radii"]),
        corners=np.asarray(pred["corners"]),
        r2_mu=np.asarray(err["r2_mu"]),
    )


# ----------------------------
# Run one model
# ----------------------------
def run_model(model_name, use_scan=True, ab_weight=1.0):
    data = get_data()
    model, params = make_model(model_name, data)
    run_training_loop(model, params, data, model_name, use_scan=use_scan, ab_weight=ab_weight)
    plot_loss(model_name)
    validate(model, data, model_name)


if __name__ == "__main__":
    run_model(LINEAR_REGRESSION, use_scan=True, ab_weight=1.0)
    run_model(VANILLA, use_scan=True, ab_weight=1.0)
    run_model(CNN, use_scan=False, ab_weight=1.0)
