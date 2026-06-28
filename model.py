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

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    # TODO: sample W from a scaled normal and set b to zeros, return as a dict.
    return {"W": scale*sample_normal_matrix(key, (in_dim, out_dim)), 
    "b": jnp.zeros(out_dim)}

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    """Builds a list of per-layer parameter dicts from adjacent layer sizes."""
    n = len(layer_sizes)

    # We need exactly (n - 1) keys since there are (n - 1) transitions between layers
    keys = jax.random.split(key, num=n-1)

    params = []
    for i in range(n - 1):
        # Initialize the layer using the designated subkey
        layer_param = init_linear_layer(
            key=keys[i],
            in_dim=layer_sizes[i],
            out_dim=layer_sizes[i + 1],
            scale=scale,
        )
        params.append(layer_param)

    return params

# Step 9 - linear_forward
def linear_forward(x, layer_params):
    # TODO: compute x @ W + b using layer_params['W'] and layer_params['b'].
    return x @ layer_params["W"] + layer_params["b"]

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    # TODO: return an array of the same shape with negatives replaced by zero.
    return jnp.maximum(0, x)

# Step 11 - softmax_probabilities
import jax.numpy as jnp


def softmax_probabilities(logits):
    """Convert logits into a numerically stable softmax along the last axis."""
    # Find the maximum value along the last axis, keeping dimensions for broadcasting
    max_logits = jnp.max(logits, axis=-1, keepdims=True)

    # Subtract the max to prevent overflow, then exponentiate
    exp_logits = jnp.exp(logits - max_logits)

    # Divide by the sum along the last axis
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

# Step 12 - mlp_forward
def mlp_forward(params, x):
    """Runs input x through all hidden layers with ReLU,

    and returns the raw logits from the final linear layer.
    """
    # Loop through all layers except the last one
    for layer in params[:-1]:
        x = linear_forward(x, layer)
        x = relu_activation(x)

    # Apply the final layer without an activation function
    logits = linear_forward(x, params[-1])

    return logits

# Step 13 - log_softmax_logits
def log_softmax_logits(logits):
    # TODO: return the numerically stable log-softmax of logits along the last axis.
    m = jnp.max(logits, axis = -1, keepdims = True)
    e = jnp.log(jnp.sum(jnp.exp(logits-m), axis = -1, keepdims = True))
    return logits - m - e

# Step 14 - cross_entropy_loss
def cross_entropy_loss(logits, one_hot_targets):
    """Computes the mean cross-entropy between predicted logits and one_hot_targets."""
    # 1. Compute stable log probabilities
    log_probs = log_softmax_logits(logits)

    # 2. Multiply element-wise by targets to mask out incorrect classes,
    # then sum across the class dimension (axis=-1) to get loss per sample
    per_sample_loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)

    # 3. Take the average loss across the entire batch
    return jnp.mean(per_sample_loss)

# Step 15 - classification_accuracy
import jax.numpy as jnp

def classification_accuracy(logits, labels):
    """Fraction of rows where argmax(logits) equals the integer label."""
    # TODO: compute predicted classes from logits and compare to labels
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)

# Step 16 - loss_fn_of_params
import jax
import jax.numpy as jnp

def loss_fn_of_params(params, x, one_hot_targets):
    # TODO: return scalar cross-entropy loss as a function of params, ready for jax.grad
    logits = mlp_forward(params, x)
    return cross_entropy_loss(logits, one_hot_targets)

# Step 17 - compute_param_grads
import jax
import jax.numpy as jnp


def compute_param_grads(params, x, one_hot_targets):
    """Computes the gradients of the cross-entropy loss with respect to params."""
    # Use jax.grad to transform the function into a gradient calculator.
    # grad_fn will have the exact same call signature as loss_fn_of_params.
    grad_fn = jax.grad(loss_fn_of_params)

    # Evaluate the gradient function on the current inputs
    return grad_fn(params, x, one_hot_targets)

# Step 18 - sgd_update_params
import jax


def sgd_update_params(params, grads, learning_rate):
    """Applies one functional SGD step to every parameter in the pytree."""
    # jax.tree.map walks through both pytrees simultaneously and applies the function to their leaves
    return jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)

# Step 19 - training_step
import jax
import jax.numpy as jnp


def training_step(params, x, one_hot_targets, learning_rate):
    """Executes a single training iteration.

    Computes current loss + grads via upstream helpers, applies a functional
    SGD update to the parameters, and returns both the new parameters and the
    loss.
    """
    # 1. Compute the loss value for tracking
    loss = loss_fn_of_params(params, x, one_hot_targets)

    # 2. Compute the gradients of the loss with respect to the parameters
    grads = compute_param_grads(params, x, one_hot_targets)

    # 3. Apply the functional SGD update to get a fresh parameter pytree
    updated_params = sgd_update_params(params, grads, learning_rate)

    # 4. Return both the updated weights and the loss scalar
    return updated_params, loss

# Step 20 - train_mlp
def train_mlp(params, x, one_hot_targets, learning_rate, num_epochs):
    """Run num_epochs full-batch SGD updates and return the final params."""
    # TODO: run num_epochs full-batch SGD updates via training_step and return final params
    for epoch in range(num_epochs):
        params, loss = training_step(params, x, one_hot_targets, learning_rate)
    return params

# Step 21 - predict_classes
def predict_classes(params, x):
    # TODO: run mlp_forward on x and return the argmax class index per row
    return jnp.argmax(mlp_forward(params, x), axis = -1)

