""" An implementation of a neural network used as autoencoder.
    Supports 4 layer types:
    fully connected, convolutional, pooling layers, transpose convolutional.

    Layer specifications and other network hyperparameters are passed to this
    program as a command line argument in form of a json file.
"""

from __future__ import print_function
from __future__ import division
from functools import reduce
import tensorflow as tf
import numpy as np
import time
import sys
import json
import os


def fully_connected_layer(previous_layer,
                          num_units,
                          activation_function,
                          init_stddev):
    """ Return a tensorflow node for a fully connected layer. """
    return tf.layers.dense(tf.layers.flatten(previous_layer), num_units,
        activation=activation_function,
        kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev))


def convolutional_layer(previous_layer,
                        num_filters,
                        filter_size,
                        stride,
                        activation_function,
                        init_stddev):
    """ Return a tensorflow node for a convolutional layer. """
    return tf.layers.conv2d(previous_layer, num_filters, filter_size, stride,
        padding="SAME", activation=activation_function,
        kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev))


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


def transpose_convolutional_layer(previous_layer,
                                  num_filters,
                                  filter_size,
                                  stride,
                                  activation_function,
                                  init_stddev):
    """ Return a tensorflow node for a transpose convolutional layer. """
    return tf.layers.conv2d_transpose(previous_layer, num_filters, filter_size,
        stride, padding="SAME", activation=activation_function,
        kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev))


class NeuralNetwork():
    """ A neural network used as an autoencoder. """

    def __init__(self, features_shape, layers, params):
        self.params = params
        # Create input layer
        x = tf.placeholder(tf.float32, (None,) + features_shape)
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
            elif layer_spec["type"] == "transpose_convolutional":
                layer_type = transpose_convolutional_layer
                layer_params.setdefault("init_stddev", params["init_stddev"])
            else:
                raise ValueError("Unknown layer type '%s'." %
                                 layer_spec["type"])
            previous_layer = layer_type(previous_layer, **layer_params)
        # Output layer (scale values to [0,1] like the input images)
        output_min = tf.reduce_min(previous_layer)
        output_max = tf.reduce_max(previous_layer)
        output = (previous_layer - output_min) / (output_max - output_min)
        # Loss function and optimizer
        if params["optimizer"] == "gd" or params["optimizer"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(
                    float(params["learning_rate"]))
        elif params["optimizer"] == "adam":
            optimizer = tf.train.AdamOptimizer(float(params["learning_rate"]))
        else:
            raise ValueError("Unknown optimization method '%s'." %
                             params["optimizer"])
        loss = tf.reduce_mean(tf.square(output - x))
        train = optimizer.minimize(loss)
        # Store important tensorflow nodes as instance attributes
        self.nodes = {
            "x": x,
            "loss": loss,
            "train": train,
            "output": output
        }
        # Start tensorflow session and initialize parameters
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.session.run(init)

    def train(self, features):
        """ Train the neural network and print evaluation results on the
            training and validation set in each iteration.
        """
        for step in range(1, self.params["max_epochs"] + 1):
            if self.params["optimizer"] == "sgd" or \
                    self.params["optimizer"] == "adam": 
                n_samples = features["train"].shape[0]
                batch_size = self.params["batch_size"]
                n_batches = n_samples // batch_size
                for _ in range(n_batches):
                    batch_indices = np.random.choice(n_samples, batch_size)
                    X_batch = features["train"][batch_indices]
                    _, train_loss = self.session.run(
                            [self.nodes["train"], self.nodes["loss"]],
                            { self.nodes["x"]: X_batch })
            elif self.params["optimizer"] == "gd":
                _, train_loss = self.session.run(
                        [self.nodes["train"], self.nodes["loss"]],
                        { self.nodes["x"]: features["train"] })
            else:
                raise ValueError("Unknown optimization method '%s'." %
                                 self.params["optimizer"])
            valid_loss = self.session.run(self.nodes["loss"],
                    { self.nodes["x"]: features["valid"] })
            print("Iteration %d, Training loss %.4f Validation Loss %.4f"
                  % (step, train_loss, valid_loss))
   
    def calculate_output(self, features):
        """ Calculate the output of the network for given input data. """
        return self.session.run(
                self.nodes["output"], { self.nodes["x"]: features })


def train_network_on_mnist(
        layers, params, subset_size=None, output_file=None, noise=None):
    """ Train a neural network defined by the given layers and parameters
        on the MNIST data set (or a subset thereof).
        Visualize the result by plotting a few random input images from the
        validation set and their corresponding outputs from the autoencoder
        next to each other into the file specified by 'output_file'.
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
    # Create a neural network, train it and measure time
    neural_network = NeuralNetwork(features_shape, layers, params)
    t0 = time.time()
    neural_network.train({ "train": X_train, "valid": X_valid })
    t1 = time.time()
    print("Duration: %.1fs" % (t1 - t0))
    # Visualize results on a few random samples of the validation set
    if output_file is not None:
        try:
            import matplotlib.pyplot as pyplot
            import matplotlib.image as imglib
        except ImportError:
            print("Module 'matplotlib' not found. Skipping visualization.")
            return
        num_samples = 40
        input_indices = np.random.choice(X_valid.shape[0], num_samples)
        inputs = X_valid[input_indices]
        # Add gaussian noise with given standard deviation
        if noise is not None:
            inputs = np.maximum(0, np.minimum(1, inputs +
                np.random.normal(scale=noise, size=inputs.shape)))
        outputs = neural_network.calculate_output(inputs)
        num_image_cols = 5
        num_image_rows = 8
        col_spacing = 40
        row_spacing = 10
        image = np.zeros((num_image_rows * (28 + row_spacing) - row_spacing,
                          num_image_cols * (2*28 + col_spacing) - col_spacing))
        for i in range(num_samples):
            col = i % num_image_cols
            row = i % num_image_rows
            x = col * (2 * 28 + col_spacing)
            y = row * (28 + row_spacing)
            image[y:y+28,x:x+28] = inputs[i].reshape((28, 28))
            image[y:y+28,x+28:x+2*28] = outputs[i].reshape((28, 28))
        imglib.imsave(output_file, image,
                format=os.path.splitext(output_file)[1][1:], cmap="Greys")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 neural_network.py <json spec-file> [subset-size]")
        sys.exit()
    filename = sys.argv[1]
    subset_size = int(sys.argv[2]) if len(sys.argv) == 3 else None
    with open(filename) as spec_file:
        network_spec = json.load(spec_file)
        train_network_on_mnist(
                network_spec["layers"], network_spec["params"], subset_size,
                output_file="visualized-results.png", noise=0.3)

