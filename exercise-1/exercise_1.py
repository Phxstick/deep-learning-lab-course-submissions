import numpy as np
import cPickle
import os
import gzip
import time


def mnist(dataset_dir="./data"):
    """ Load the MNIST data from given directory (download it if necessary). """
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    data_file = os.path.join(dataset_dir, "mnist.pkl.gz")
    if not os.path.exists(data_file):
        print("... downloading MNIST from the web")
        try:
            import urllib
            urllib.urlretrieve("http://google.com")
        except AttributeError:
            import urllib.request as urllib
        url = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"
        urllib.urlretrieve(url, data_file)

    print("Loading data ...")
    # Load the dataset
    f = gzip.open(data_file, "rb")
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype("float32")
    test_x = test_x.reshape(test_x.shape[0], 1, 28, 28)
    test_y = test_y.astype("int32")
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype("float32")
    valid_x = valid_x.reshape(valid_x.shape[0], 1, 28, 28)
    valid_y = valid_y.astype("int32")
    train_x, train_y = train_set
    train_x = train_x.astype("float32").reshape(train_x.shape[0], 1, 28, 28)
    train_y = train_y.astype("int32")
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    print("Done.")
    return rval


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_d(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def tanh(x):
    return np.tanh(x)

def tanh_d(x):
    return 1 - np.square(np.tanh(x))

def relu(x):
    return np.maximum(0.0, x)

def relu_d(x):
    return (x >= 0).astype(float)

def softmax(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_safe = x - x_max
    e_x = np.exp(x_safe)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def one_hot(labels):
    """ This creates a one hot encoding from a flat vector.
    E.g. given [0,2,1] it creates [[1,0,0], [0,0,1], [0,1,0]].
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels

def unhot(one_hot_labels):
    """ Invert a one hot encoding, creating a flat vector. """
    return np.argmax(one_hot_labels, axis=-1)


class Activation(object):

    def __init__(self, tname):
        if tname == "sigmoid":
            self.act = sigmoid
            self.act_d = sigmoid_d
        elif tname == "tanh":
            self.act = tanh
            self.act_d = tanh_d
        elif tname == "relu":
            self.act = relu
            self.act_d = relu_d
        else:
            raise ValueError("Invalid activation function.")

    def fprop(self, input):
        # Cache last input to calculate derivative for backpropagation
        self.last_input = input
        return self.act(input)

    def bprop(self, output_grad):
        return output_grad * self.act_d(self.last_input)


class Layer(object):

    def fprop(self, input):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError("This is an interface class, please use a "
                                  "derived instance")

    def bprop(self, output_grad):
        """ Calculate input gradient and gradient with respect to weights and
        bias (backpropagation). """
        raise NotImplementedError("This is an interface class, please use a "
                                  "derived instance")

    def output_size(self):
        """ Calculate size of this layer's output.
        input_shape[0] is the number of samples in the input.
        input_shape[1] is the shape of the feature.
        """
        raise NotImplementedError("This is an interface class, please use a "
                                  "derived instance")


class Loss(object):

    def loss(self, output, output_net):
        """ Calculate mean loss given real output and network output. """
        raise NotImplementedError("This is an interface class, please use a "
                                  "derived instance")

    def input_grad(self, output, output_net):
        """ Calculate input gradient given real output and network output. """
        raise NotImplementedError("This is an interface class, please use a "
                                  "derived instance")


class Parameterized(object):

    def params(self):
        """ Return parameters. """
        raise NotImplementedError("This is an interface class, please use a "
                                  "derived instance")

    def grad_params(self):
        """ Return gradient with respect to parameters. """
        raise NotImplementedError("This is an interface class, please use a "
                                  "derived instance")
    
    def set_params(self):
        """ Set parameters. """
        raise NotImplementedError("This is an interface class, please use a "
                                  "derived instance")


class InputLayer(Layer):
    
    def __init__(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError("InputLayer requires input_shape as a tuple")
        self.input_shape = input_shape

    def output_size(self):
        return self.input_shape

    def fprop(self, input):
        return input

    def bprop(self, output_grad):
        return output_grad


class FullyConnectedLayer(Layer, Parameterized):
    """ A standard fully connected hidden layer, as discussed in the lecture.
    """

    def __init__ (self, input_layer, num_units, init_stddev,
                  activation_fun=Activation("relu")):
        self.num_units = num_units
        self.activation_fun = activation_fun
        # input shape has the form (batch_size, num_units_prev)
        # where num_units_prev is the number of units in the input layer
        self.input_shape = input_layer.output_size()
        # weight matrix of shape (num_units_prev, num_units)
        self.W = np.random.normal(size=(self.input_shape[1], num_units),
                                  scale=init_stddev)
        # bias vector of shape (num_units)
        self.b = np.random.normal(size=num_units, scale=init_stddev)
        # dummy variables for parameter gradients
        self.dW = None
        self.db = None

    def output_size(self):
        return (self.input_shape[0], self.num_units)

    def fprop(self, input):
        # Cache last input to calculate derivative for backpropagation
        self.last_input = input
        result = np.dot(input, self.W) + self.b
        if self.activation_fun is not None:
            result = self.activation_fun.fprop(result)
        return result

    def bprop(self, output_grad):
        if self.activation_fun is not None:
            output_grad = self.activation_fun.bprop(output_grad)
        self.db = np.sum(output_grad, axis=0)
        self.dW = np.dot(np.transpose(self.last_input), output_grad)
        grad_input = np.dot(output_grad, np.transpose(self.W))
        return grad_input

    def params(self):
        return self.W, self.b

    def grad_params(self):
        return self.dW, self.db

    def set_params(self, W, b):
        self.W, self.b = W, b


class LinearOutput(Layer, Loss):
    """ A simple linear output layer that uses a squared loss (for regression).
    """

    def __init__(self, input_layer):
        self.input_size = input_layer.output_size()

    def output_size(self):
        return (1,)

    def fprop(self, input):
        return input

    def bprop(self, output_grad):
        raise NotImplementedError("bprop cannot be called for an output layer.")

    def input_grad(self, Y, Y_pred):
        return Y_pred - Y

    def loss(self, Y, Y_pred):
        loss = 0.5 * np.square(Y - Y_pred)
        return np.mean(np.sum(loss, axis=1))


class SoftmaxOutput(Layer, Loss):
    """ A softmax output layer that calculates the negative log likelihood as
    loss and should be used for classification.
    """

    def __init__(self, input_layer):
        self.input_size = input_layer.output_size()

    def output_size(self):
        return (1,)

    def fprop(self, input):
        return softmax(input)

    def bprop(self, output_grad):
        raise NotImplementedError("bprop cannot be called for an output layer.")

    def input_grad(self, Y, Y_pred):
        return Y_pred - Y

    def loss(self, Y, Y_pred):
        out = softmax(Y_pred)
        # Add epsilon in the log to make loss numerically stable
        eps = 1e-10
        # Assume one-hot encoding of Y
        loss = -np.log(np.max(Y * out, axis=1) + eps)
        return np.mean(loss)


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def _loss(self, X, Y):
        # Assume one-hot encoding of Y
        Y_pred = self.predict(X)
        return self.layers[-1].loss(Y, Y_pred)
    
    def predict(self, X):
        """ Calculate an output Y for the given input X (forward propagation).
        """
        Y_pred = X
        for layer in self.layers:
            Y_pred = layer.fprop(Y_pred)
        return Y_pred

    def backpropagate(self, Y, Y_pred, upto=0):
        """ Backpropagation of partial derivatives through the complete network
        up to layer 'upto'.
        """
        next_grad = self.layers[-1].input_grad(Y, Y_pred)
        for i in range(len(self.layers) - 2, upto - 1, -1):
            next_grad = self.layers[i].bprop(next_grad)
        return next_grad

    def classification_error(self, X, Y):
        """ Calculate error on the given data assuming they are classes that
        should be predicted (not one-hot encoded).
        """
        Y_pred = unhot(self.predict(X))
        error = Y_pred != Y
        return np.mean(error)

    def sgd_epoch(self, X, Y, learning_rate, batch_size):
        """ Perform stochastic gradient descent to update network parameters.
        """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        for b in range(n_batches):
            batch_indices = np.random.choice(X.shape[0], batch_size)
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]
            Y_pred = self.predict(X_batch)
            self.backpropagate(Y_batch, Y_pred)
            # TODO: Make learning rate go to zero? How fast?
            for i in range(1, len(self.layers) - 1):
                updated_params = tuple(map(lambda p, g: p - learning_rate * g,
                    zip(self.layers[i].params(), self.layers[i].grad_params())))
                self.layers[i].set_params(*updated_params)

    def gd_epoch(self, X, Y, learning_rate):
        """ Perform gradient descent to update network parameters.
        """
        Y_pred = self.predict(X)
        self.backpropagate(Y, Y_pred)
        for i in range(1, len(self.layers) - 1):
            updated_params = tuple(map(lambda (p, g): p - learning_rate * g,
                zip(self.layers[i].params(), self.layers[i].grad_params())))
            self.layers[i].set_params(*updated_params)

    def train(self, X, Y, X_valid, Y_valid, learning_rate=0.1, max_epochs=100,
              batch_size=64, descent_type="sgd", y_one_hot=True):
        """ Train network on the given data. """
        if y_one_hot:
            Y_train = one_hot(Y)
        else:
            Y_train = Y
        print("... starting training")
        for e in range(max_epochs + 1):
            if descent_type == "sgd":
                self.sgd_epoch(X, Y_train, learning_rate, batch_size)
            elif descent_type == "gd":
                self.gd_epoch(X, Y_train, learning_rate)
            else:
                raise NotImplementedError(
                    "Unknown gradient descent type {}".format(descent_type))

            # Output error on the training and validation set
            train_loss = self._loss(X, Y_train)
            train_error = self.classification_error(X, Y)
            valid_error = self.classification_error(X_valid, Y_valid)
            print("epoch {:.4f}, loss {:.4f}, train error {:.4f}, "
                  "validation error {:.4f}"
                  .format(e, train_loss, train_error, valid_error))

    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for correctness.
        """
        for l, layer in enumerate(self.layers):
            if isinstance(layer, Parameterized):
                print('checking gradient for layer {}'.format(l))
                for p, param in enumerate(layer.params()):
                    param_shape = param.shape

                    def output_given_params(param_new):
                        """ A function that will compute the output of
                            the network given a set of parameters.
                        """
                        param[:] = np.reshape(param_new, param_shape)
                        return self._loss(X, Y)
                   
                    def grad_given_params(param_new):
                        """ A function that will compute the gradient of the
                            network given a set of parameters
                        """
                        param[:] = np.reshape(param_new, param_shape)
                        Y_pred = self.predict(X)
                        self.backpropagate(Y, Y_pred, upto=l)
                        return np.ravel(self.layers[l].grad_params()[p])
                   
                    param_init = np.ravel(np.copy(param))
                    epsilon = 1e-4
                    gparam_fd = np.zeros_like(param_init)
                    gparam_bprop = np.zeros_like(param_init)

                    for i in range(param_init.shape[0]):
                        param_init[i] += epsilon
                        gparam_fd[i] = output_given_params(param_init)
                        param_init[i] -= epsilon
                        gparam_fd[i] -= output_given_params(param_init)
                        gparam_fd[i] /= epsilon
                        gparam_bprop[i] = grad_given_params(param_init)[i]

                    # print("gparam_fd:    ", gparam_fd)
                    # print("gparam_bprop: ", gparam_bprop)
                    err_mean = np.mean(np.abs(gparam_bprop - gparam_fd))
                    err_l2 = np.sqrt(np.sum(np.square(gparam_bprop - gparam_fd)))
                    print("diff (mean)    {:.2e}".format(err_mean))
                    print("diff (l2-norm) {:.2e}".format(err_l2))

                    # import scipy.optimize
                    # err_scipy = scipy.optimize.check_grad(
                    #         output_given_params, grad_given_params, param_init,
                    #         epsilon=epsilon)
                    # print("diff (scipy)    {:.2e}".format(err_scipy))
                    # print

                    # assert(err_mean < epsilon)

                    # reset parameters
                    param[:] = np.reshape(param_init, param_shape)


def check_gradient_on_random_data():
    input_shape = (50, 10)
    n_labels = 6

    layers = [InputLayer(input_shape)]
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=15,
            init_stddev=0.1,
            activation_fun=Activation("relu")
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=n_labels,
            init_stddev=0.1,
            activation_fun=Activation("relu")
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=n_labels,
            init_stddev=0.1,
            activation_fun=Activation("tanh")
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=n_labels,
            init_stddev=0.1,
            activation_fun=Activation("relu")
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=n_labels,
            init_stddev=0.1,
            activation_fun=Activation("relu")
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=n_labels,
            init_stddev=0.1,
            activation_fun=Activation("relu")
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=n_labels,
            init_stddev=0.1,
            activation_fun=None
    ))
    layers.append(SoftmaxOutput(layers[-1]))
    nn = NeuralNetwork(layers)

    X = np.random.normal(size=input_shape)
    y = np.zeros((input_shape[0], n_labels))
    for i in range(y.shape[0]):
        idx = np.random.randint(n_labels)
        y[i, idx] = 1.

    nn.check_gradients(X, y)

    # y = unhot(y)
    # valid_shape = (20, 10)
    # X_valid = np.random.normal(size=valid_shape)
    # y_valid = np.zeros((valid_shape[0], n_labels))
    # for i in range(y_valid.shape[0]):
    #     idx = np.random.randint(n_labels)
    #     y_valid[i, idx] = 1.
    # y_valid = unhot(y_valid)

    # nn.train(X, y, X_valid, y_valid, learning_rate=0.1,
    #          max_epochs=20, batch_size=64, descent_type="gd", y_one_hot=True)


def evaluate_on_mnist(random_subset=False):
    d_train, d_val, d_test = mnist()
    X_train, y_train = d_train
    X_valid, y_valid = d_val

    if random_subset:
        # Downsample data to make it a bit faster
        n_train_samples = 10000
        train_idxs = np.random.permutation(X_train.shape[0])[:n_train_samples]
        X_train = X_train[train_idxs]
        y_train = y_train[train_idxs]

    print("X_train shape: {}".format(np.shape(X_train)))
    print("y_train shape: {}".format(np.shape(y_train)))
    X_train = X_train.reshape((X_train.shape[0], -1))
    print("Reshaped X_train size: {}".format(X_train.shape))
    X_valid = X_valid.reshape((X_valid.shape[0], -1))
    print("Reshaped X_valid size: {}".format(X_valid.shape))

    input_shape = (None, 28 * 28)
    init_stddev = 0.01
    layers = [InputLayer(input_shape)]
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=10,
            init_stddev=init_stddev,
            activation_fun=Activation("relu")
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=10,
            init_stddev=init_stddev,
            activation_fun=Activation("relu")
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=10,
            init_stddev=init_stddev,
            activation_fun=Activation("relu")
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=10,
            init_stddev=init_stddev,
            activation_fun=Activation("relu")
    ))
    layers.append(FullyConnectedLayer(
            layers[-1],
            num_units=10,
            init_stddev=init_stddev,
            activation_fun=None
    ))
    layers.append(SoftmaxOutput(layers[-1]))
    nn = NeuralNetwork(layers)

    t0 = time.time()
    nn.train(X_train, y_train, X_valid, y_valid, learning_rate=0.1,
             max_epochs=20, batch_size=64, descent_type="gd", y_one_hot=True)
    t1 = time.time()
    print("Duration: {:.1f}s".format(t1-t0))


if __name__ == "__main__":
    check_gradient_on_random_data()
    # evaluate_on_mnist(True)

