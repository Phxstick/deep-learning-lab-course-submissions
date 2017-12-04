import os
import sys
import json
import time
import h5py
import random
import numpy as np
from neural_network import NeuralNetwork

class Data:

    def __init__(self):
        """ Load data from the HDF5 file. """
        with h5py.File("cell_data.h5", "r") as data:
            self.train_images = \
                    [data["/train_image_{}".format(i)][:] for i in range(28)]
            self.train_labels = \
                    [data["/train_label_{}".format(i)][:] for i in range(28)]
            self.test_images = \
                    [data["/test_image_{}".format(i)][:] for i in range(3)]
            self.test_labels = \
                    [data["/test_label_{}".format(i)][:] for i in range(3)]
        self.input_resolution = 300
        self.label_resolution = 116
        self.offset = (300 - 116) // 2
  
    def get_train_image_list_and_label_list(self):
        """ Return training images and labels of size self.input_resolution. """
        n = random.randint(0, len(self.train_images) - 1)
        x = random.randint(0,
                (self.train_images[n].shape)[1] - self.input_resolution - 1)
        y = random.randint(0,
                (self.train_images[n].shape)[0] - self.input_resolution - 1)
        image = self.train_images[n][y:y + self.input_resolution,
                                     x:x + self.input_resolution, :]
        x += self.offset
        y += self.offset
        label = self.train_labels[n][y:y + self.label_resolution,
                                     x:x + self.label_resolution]
        return np.stack([image]), np.stack([label])
  
    def get_test_image_list_and_label_list(self):
        """ Return test images and labels of size self.input_resolution. """
        coord_list = [[0,0], [0, 116], [0, 232], 
                      [116,0], [116, 116], [116, 232],
                      [219,0], [219, 116], [219, 232]]
        image_list = []
        label_list = []
        for image_id in range(3):
            for y, x in coord_list:
                image = self.test_images[image_id][y:y+self.input_resolution,
                                                   x:x+self.input_resolution, :]
                image_list.append(image)
                x += self.offset
                y += self.offset
                label = self.test_labels[image_id][y:y + self.label_resolution,
                                                   x:x + self.label_resolution]
                label_list.append(label)
        return np.stack(image_list), np.stack(label_list)


def train_network(layers, params, output_filename=None):
    """ Train a neural network defined by the given layers and parameters
        on random cell segmentation training examples.
        Visualize the segmentation result for random examples from the
        validation set and plot them into the file specified by given name.
    """
    data = Data()
    X_train, y_train = data.get_train_image_list_and_label_list()
    X_valid, y_valid = data.get_test_image_list_and_label_list()
    features_shape = (300, 300, 1)
    labels_shape = (116, 116)
    num_labels = 2
    # Create a neural network, train it and measure time
    neural_network = NeuralNetwork(
            features_shape, labels_shape, num_labels, layers, params)
    t0 = time.time()
    neural_network.train({ "train": X_train, "valid": X_valid },
                         { "train": y_train, "valid": y_valid })
    t1 = time.time()
    print("Duration: %.1fs" % (t1 - t0))
    # Visualize results on a few random samples of the validation set
    if output_filename is not None:
        try:
            import matplotlib.pyplot as pyplot
            import matplotlib.image as imglib
        except ImportError:
            print("Module 'matplotlib' not found. Skipping visualization.")
            return
        num_samples = 3
        input_indices = np.random.choice(X_valid.shape[0], num_samples)
        inputs = X_valid[input_indices]
        labels = -y_valid[input_indices] + 1
        outputs = labels  # -np.argmax(neural_network.predict(inputs), axis=-1) + 1
        y_spacing = 20
        x_spacing = 10
        x_size, y_size = labels_shape
        x_size_input, y_size_input = features_shape[:2]
        x_offset = int((x_size_input - x_size) / 2)
        y_offset = int((y_size_input - y_size) / 2)
        cropped_inputs = inputs[:, x_offset : x_offset + x_size,
                                   y_offset : y_offset + y_size, 0]
        output_image = np.zeros(
            (num_samples * (y_size + y_spacing) - y_spacing,
             3 * x_size + 2 * x_spacing))
        for i in range(num_samples):
            y = i * (y_size + y_spacing)
            x1 = 0
            x2 = x_size + x_spacing
            x3 = 2 * x_size + 2 * x_spacing
            output_image[y:y+y_size,x1:x1+x_size] = cropped_inputs[i]
            output_image[y:y+y_size,x2:x2+x_size] = labels[i]
            output_image[y:y+y_size,x3:x3+x_size] = outputs[i]
        imglib.imsave(output_filename, output_image,
                format=os.path.splitext(output_filename)[1][1:], cmap="Greys")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 main.py <network spec file> [output file]")
        sys.exit()
    spec_filename = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) == 3 else None
    with open(spec_filename) as spec_file:
        network_spec = json.load(spec_file)
        train_network(network_spec["layers"], network_spec["params"],
                      output_filename)

