"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed: int) -> jax.Array:
    """Wraps a Python integer seed into a JAX PRNG key (uint32 array of shape (2,))."""
    return jax.random.PRNGKey(seed)

# Step 2 - split_prng_key
import jax


def split_prng_key(key, num: int):
    """Splits a JAX PRNG key into `num` independent subkeys.

    Returns a (num, 2) array of dtype uint32.
    """
    return jax.random.split(key, num=num)

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    # TODO: return a jnp array of the given shape with i.i.d. N(0,1) samples drawn from key
    return jax.random.normal(key, shape)

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    # TODO: draw a batch of random input feature vectors from the PRNG key
    return sample_normal_matrix(key, (batch_size, num_features))

# Step 5 - assign_class_labels
import jax.numpy as jnp


def assign_class_labels(inputs, num_classes):
    """Turns a feature matrix into a vector of integer class labels

    by taking the argmax over the first num_classes columns.
    """
    # Slice the inputs to keep only the first num_classes columns: shape (batch_size, num_classes)
    relevant_features = inputs[:, :num_classes]

    # Take the argmax across the columns (axis=1) and cast to int32
    return jnp.argmax(relevant_features, axis=1).astype(jnp.int32)

# Step 6 - one_hot_encode_labels
import jax
import jax.numpy as jnp


def one_hot_encode_labels(labels, num_classes):
    """Convert a 1-D array of integer class indices into a 2-D one-hot matrix

    of shape (batch, num_classes) with a float32 dtype.
    """
    return jax.nn.one_hot(labels, num_classes=num_classes, dtype=jnp.float32)

# Step 7 - init_linear_layer (not yet solved)
# TODO: implement

# Step 8 - init_mlp_params (not yet solved)
# TODO: implement

# Step 9 - linear_forward (not yet solved)
# TODO: implement

# Step 10 - relu_activation (not yet solved)
# TODO: implement

# Step 11 - softmax_probabilities (not yet solved)
# TODO: implement

# Step 12 - mlp_forward (not yet solved)
# TODO: implement

# Step 13 - log_softmax_logits (not yet solved)
# TODO: implement

# Step 14 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 15 - classification_accuracy (not yet solved)
# TODO: implement

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

