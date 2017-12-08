""" An implementation of a neural network used for image segmentation.
    Supports 4 layer types:
    fully connected, convolutional, pooling layers, transpose convolutional.

    Layer specifications and other network hyperparameters are passed to the
    class constructor as JSON objects.
"""

import tensorflow as tf
import numpy as np


def fully_connected_layer(previous_layer,
                          num_units,
                          activation_function):
    """ Return a tensorflow node for a fully connected layer. """
    return tf.layers.dense(tf.layers.flatten(previous_layer), num_units,
        activation=activation_function)


def convolutional_layer(previous_layer,
                        num_filters,
                        filter_size,
                        stride,
                        activation_function):
    """ Return a tensorflow node for a convolutional layer. """
    return tf.layers.conv2d(previous_layer, num_filters, filter_size, stride,
        padding="VALID", activation=activation_function)


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
                                  activation_function):
    """ Return a tensorflow node for a transpose convolutional layer. """
    return tf.layers.conv2d_transpose(previous_layer, num_filters, filter_size,
        stride, padding="VALID", activation=activation_function)


class NeuralNetwork():
    """ A neural network used for image segmentation. """

    def __init__(self, features_shape, labels_shape, num_labels, layers, params):
        self.params = params
        # Create input layer
        x = tf.placeholder(tf.float32, (None,) + features_shape)
        y = tf.placeholder(tf.int32, (None,) + labels_shape)
        y_one_hot = tf.one_hot(y, num_labels)
        previous_layer = x
        # Create hidden layers according to given specification
        labeled_layers = dict()
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
            elif layer_spec["type"] == "convolutional":
                layer_type = convolutional_layer
            elif layer_spec["type"] == "pooling":
                layer_type = pooling_layer
            elif layer_spec["type"] == "transpose_convolutional":
                layer_type = transpose_convolutional_layer
            else:
                raise ValueError("Unknown layer type '%s'." %
                                 layer_spec["type"])
            if "reuse_features_from" in layer_params:
                features = labeled_layers[layer_params["reuse_features_from"]]
                x_size_old, y_size_old = features.shape.as_list()[1:3]
                x_size, y_size = previous_layer.shape.as_list()[1:3]
                x_offset = int((x_size_old - x_size) / 2)
                y_offset = int((y_size_old - y_size) / 2)
                cropped_features = tf.slice(features,
                        [0, x_offset, y_offset, 0], [-1, x_size, y_size, -1])
                previous_layer = tf.concat(
                        [cropped_features, previous_layer], axis=-1)
            previous_layer = layer_type(previous_layer, **layer_params)
            if "id" in layer_params:
                labeled_layers[layer_params["id"]] = previous_layer
        # Output layer, loss and intersection-over-union accuracy
        output = tf.nn.softmax(previous_layer)
        epsilon = 1e-8
        loss = tf.reduce_mean(-tf.log(tf.reduce_max(output*y_one_hot, axis=-1) +
                                      epsilon))
        predictions = tf.argmax(output, axis=-1, output_type=tf.int32)
        num_correct_predictions = \
                tf.reduce_sum(tf.cast(tf.equal(predictions, y), tf.float32))
        num_pixels = tf.cast(tf.size(predictions, out_type=tf.int32),tf.float32)
        accuracy = num_correct_predictions / (2 * num_pixels -
                                              num_correct_predictions)
        # Create optimizer
        if params["optimizer"] == "gd" or params["optimizer"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(
                    params["learning_rate"])
        elif params["optimizer"] == "adam":
            optimizer = tf.train.AdamOptimizer(
                    params["learning_rate"], params["beta1"], params["beta2"])
        else:
            raise ValueError("Unknown optimization method '%s'." %
                             params["optimizer"])
        train = optimizer.minimize(loss)
        # Store important tensorflow nodes as instance attributes
        self.nodes = {
            "x": x,
            "y": y,
            "loss": loss,
            "train": train,
            "accuracy": accuracy,
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
            if self.params["optimizer"] == "sgd" or \
                    self.params["optimizer"] == "adam": 
                n_samples = features["train"].shape[0]
                batch_size = self.params["batch_size"]
                n_batches = n_samples // batch_size
                for _ in range(n_batches):
                    batch_indices = np.random.choice(n_samples, batch_size)
                    X_batch = features["train"][batch_indices]
                    y_batch = labels["train"][batch_indices]
                    _, train_loss, train_accuracy = self.session.run(
                            [self.nodes["train"], self.nodes["loss"],
                             self.nodes["accuracy"]],
                            { self.nodes["x"]: X_batch,
                              self.nodes["y"]: y_batch })
            elif self.params["optimizer"] == "gd":
                _, train_loss, train_accuracy = self.session.run(
                        [self.nodes["train"], self.nodes["loss"],
                         self.nodes["accuracy"]],
                        { self.nodes["x"]: features["train"],
                          self.nodes["y"]: labels["train"] })
            else:
                raise ValueError("Unknown optimization method '%s'." %
                                 self.params["optimizer"])
            valid_accuracy = self.session.run(self.nodes["accuracy"],
                    { self.nodes["x"]: features["valid"],
                      self.nodes["y"]: labels["valid"] })
            print("Iteration %d, Training loss %.4f, Training accuracy %.4f, "
                  "Validation accuracy %.4f"
                  % (step, train_loss, train_accuracy, valid_accuracy))
   
    def predict(self, features):
        """ Segment given input data. """
        return self.session.run(
                self.nodes["predictions"], { self.nodes["x"]: features })

