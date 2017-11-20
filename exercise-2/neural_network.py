""" An implementation of a neural network for classification tasks.
    Supports 3 layer types: fully connected, convolutional, pooling layers.

    Layer specifications and other network hyperparameters are passed to this
    program as a command line argument in form of a json file.
    The output layer should not be included in the specification -- It is always
    assumed to be a fully connected layer applying the softmax function.
"""

from __future__ import print_function
from __future__ import division
from functools import reduce
import tensorflow as tf
import numpy as np
import time
import sys
import json


def fully_connected_layer(previous_layer,
                          num_units,
                          activation_function,
                          init_stddev):
    """ Return a tensorflow node for a fully connected layer. """
    num_features_prev = int(reduce(lambda d1, d2: d1 * d2,
                                   tuple(previous_layer.shape[1:])))
    input_flattened = tf.reshape(previous_layer, (-1, num_features_prev))
    normal_dist = tf.distributions.Normal(0.0, init_stddev)
    W_shape = (num_features_prev, num_units)
    b_shape = (num_units,)
    W = tf.Variable(initial_value=normal_dist.sample(W_shape),
                    expected_shape=W_shape, dtype=tf.float32)
    b = tf.Variable(initial_value=normal_dist.sample(b_shape),
                    expected_shape=b_shape, dtype=tf.float32)
    result = tf.matmul(input_flattened, W) + b
    if activation_function is not None:
        result = activation_function(result)
    return result


def convolutional_layer(previous_layer,
                        num_filters,
                        filter_size,
                        stride,
                        padding,
                        activation_function,
                        init_stddev):
    """ Return a tensorflow node for a convolutional layer. """
    normal_dist = tf.distributions.Normal(0.0, init_stddev)
    num_channels_prev = int(previous_layer.shape[-1])
    filter_shape = (filter_size, filter_size, num_channels_prev, num_filters)
    filters = tf.Variable(initial_value=normal_dist.sample(filter_shape),
                          expected_shape=filter_shape, dtype=tf.float32)
    feature_shape_prev = tuple(previous_layer.shape[1:-1])
    padded_input = tf.pad(previous_layer, tf.constant(
        ((0,0),) + ((padding, padding),) * len(feature_shape_prev) + ((0,0),)))
    result = tf.nn.convolution(padded_input, filters, padding="VALID",
            strides=(stride,) * len(feature_shape_prev), data_format="NHWC")
    bias_shape = (1,) + (1,) * len(feature_shape_prev) + (num_filters,)
    bias = tf.Variable(initial_value=normal_dist.sample(bias_shape),
                       expected_shape=bias_shape, dtype=tf.float32)
    result = result + bias
    if activation_function is not None:
        result = activation_function(result)
    return result


def pooling_layer(previous_layer,
                  window_size,
                  stride,
                  pooling_type):
    """ Return a tensorflow node for a pooling layer. """
    feature_shape_prev = tuple(previous_layer.shape[1:-1])
    window_shape = (window_size,) * len(feature_shape_prev)
    strides = (stride,) * len(feature_shape_prev)
    result = tf.nn.pool(previous_layer, window_shape, pooling_type,
            padding="VALID", strides=strides, data_format="NHWC")
    return result


class NeuralNetwork():
    """ A neural network used for classifaction (using softmax output). """

    def __init__(self, features_shape, num_labels, layers, params):
        self.params = params
        # Create input layer (assume one-hot encoding of labels)
        x = tf.placeholder(tf.float32, (None,) + features_shape)
        y_one_hot = tf.placeholder(tf.float32, (None, num_labels))
        y = tf.argmax(y_one_hot, axis=1)
        previous_layer = x
        # Create hidden layers according to given specification
        for layer_spec in layers:
            layer_params = layer_spec["params"]
            if "activation_function" in layer_params:
                if layer_params["activation_function"] == "relu":
                    layer_params["activation_function"] = tf.nn.relu
                elif layer_params["activation_function"] == "tanh":
                    layer_params["activation_function"] = tf.nn.tanh
                elif layer_params["activation_function"] is not None:
                    raise ValueError("Unknown activation function '%s'." %
                                     layer_params["activation_function"])
            layer_type = None
            if layer_spec["type"] == "fully_connected":
                layer_type = fully_connected_layer
                layer_params.setdefault("init_stddev", params["init_stddev"])
            elif layer_spec["type"] == "convolutional":
                layer_type = convolutional_layer
                layer_params.setdefault("init_stddev", params["init_stddev"])
            elif layer_spec["type"] == "pooling":
                layer_type = pooling_layer
            else:
                raise ValueError("Unknown layer type '%s'." %
                                 layer_spec["type"])
            previous_layer = layer_type(previous_layer, **layer_params)
        # Create output layer, calculate loss and classification error
        output_layer = fully_connected_layer(previous_layer, num_labels,
                activation_function=None, init_stddev=params["init_stddev"])
        output = tf.nn.softmax(output_layer)
        loss = tf.reduce_mean(-tf.log(tf.reduce_max(output * y_one_hot, axis=1)))
        predictions = tf.argmax(output, axis=1)
        classification_error = 1 - tf.reduce_mean(
                tf.cast(tf.equal(predictions, y), tf.float32))
        # Create optimizer
        if params["optimizer"] == "gd" or params["optimizer"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(
                    float(params["learning_rate"]))
        else:
            raise ValueError("Unknown optimization method '%s'." %
                             params["optimizer"])
        train = optimizer.minimize(loss)
        # Store important tensorflow nodes as instance attributes
        self.nodes = {
            "x": x,
            "y": y_one_hot,
            "loss": loss,
            "train": train,
            "error": classification_error,
            "predictions": predictions
        }
        # Start tensorflow session and initialize parameters
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.session.run(init)

    def train(self, features, labels):
        """ Train the neural network and print evaluation results on the
            training and validation set in each iteration.
        """
        for step in range(1, self.params["max_epochs"] + 1):
            if self.params["optimizer"] == "sgd": 
                n_samples = features["train"].shape[0]
                batch_size = self.params["batch_size"]
                n_batches = n_samples // batch_size
                for _ in range(n_batches):
                    batch_indices = np.random.choice(n_samples, batch_size)
                    X_batch = features["train"][batch_indices]
                    y_batch = labels["train"][batch_indices]
                    _, train_loss, train_error = self.session.run(
                            [self.nodes["train"], self.nodes["loss"],
                             self.nodes["error"]],
                            { self.nodes["x"]: X_batch,
                              self.nodes["y"]: y_batch })
            elif self.params["optimizer"] == "gd":
                _, train_loss, train_error = self.session.run(
                        [self.nodes["train"], self.nodes["loss"],
                         self.nodes["error"]],
                        { self.nodes["x"]: features["train"],
                          self.nodes["y"]: labels["train"] })
            else:
                raise ValueError("Unknown optimization method '%s'." %
                                 self.params["optimizer"])
            valid_error = self.session.run(self.nodes["error"],
                    { self.nodes["x"]: features["valid"],
                      self.nodes["y"]: labels["valid"] })
            print("Iteration %d, Training loss %.4f, Training error %.4f, "
                  "Validation Error %.4f"
                  % (step, train_loss, train_error, valid_error))


def train_network_on_mnist(layers, params, subset_size=None):
    """ Train a neural network defined by the given layers and parameters
        on the MNIST data set (or a subset thereof).
    """
    # Load MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist-data", one_hot=True)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_valid, y_valid = mnist.validation.images, mnist.validation.labels
    if subset_size is not None:
        X_train, y_train = X_train[-subset_size:], y_train[-subset_size:]
        X_valid, y_valid = X_valid[-subset_size:], y_valid[-subset_size:]
    X_train = np.reshape(X_train, (-1, 28, 28, 1))
    X_valid = np.reshape(X_valid, (-1, 28, 28, 1))
    features_shape = (28, 28, 1)
    num_labels = 10
    # Create a neural network, train it and measure time
    neural_network = NeuralNetwork(features_shape, num_labels, layers, params)
    t0 = time.time()
    neural_network.train({ "train": X_train, "valid": X_valid },
                         { "train": y_train, "valid": y_valid })
    t1 = time.time()
    print("Duration: %.1fs" % (t1 - t0))


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 neural_network.py <json spec-file> [subset-size]")
        sys.exit()
    filename = sys.argv[1]
    subset_size = int(sys.argv[2]) if len(sys.argv) == 3 else None
    with open(filename) as spec_file:
        network_spec = json.load(spec_file)
        train_network_on_mnist(
                network_spec["layers"], network_spec["params"], subset_size)

