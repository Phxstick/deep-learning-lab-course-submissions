import pickle
import numpy as np
from robo import fmin


rf = pickle.load(open("./rf_surrogate_cnn.pkl", "rb"))
cost_rf = pickle.load(open("./rf_cost_surrogate_cnn.pkl", "rb"))

# Lower and upper bounds for each hyperparameter
hyperparam_ranges = [(-6,0), (32,512), (4, 10), (4, 10), (4, 10)]


def normalize_hyperparameters(x):
    """ Return numpy array of given hyperparameters normalized to [0,1]. """
    x_norm = np.copy(x)
    for i in range(len(x_norm)):
        lower_bound, upper_bound = hyperparam_ranges[i]
        if not lower_bound <= x[i] <= upper_bound:
            raise ValueError("Value x[%d] is %d, should be in [%d, %d]"
                    % (i, x[i], lower_bound, upper_bound))
        x_norm[i] = (x_norm[i] - lower_bound) / (upper_bound - lower_bound)
    return x_norm


def objective_function(x, epoch=40):
    """
    Function wrapper to approximate the validation error of the hyperparameter
    configurations x by the prediction of a surrogate regression model, which
    was trained on the validation error of randomly sampled hyperparameter
    configurations.
    The original surrogate predicts the validation error after a given epoch.
    Since all hyperparameter configurations were trained for a total amount of
    40 epochs, we will query the performance after epoch 40.
    """
    x_norm = normalize_hyperparameters(x)
    x_norm = np.append(x_norm, epoch)
    y = rf.predict(x_norm[None, :])[0]
    return y


def runtime(x, epoch=40):
    """
    Function wrapper to approximate the runtime of the hyperparameter
    configurations x.
    """
    x_norm = normalize_hyperparameters(x)
    x_norm = np.append(x_norm, epoch)
    y = cost_rf.predict(x_norm[None, :])[0]
    return y


def random_search(num_iterations=50, num_runs=10, output_filename=None):
    """
    Run random search with the given number of iterations to find a good
    hyperparameter configuration on the network surrogate.
    Run the whole search a given number of times and write the average
    optimization trajectory (incumbent errors per epoch) as well as the
    cumulative runtime of all configuration evaluations per epoch into
    the file specified the by given filename.
    """
    incumbent_errors = np.zeros(num_iterations)
    cumulative_runtimes = np.zeros(num_iterations)
    for _ in range(num_runs):
        incumbent = None
        incumbent_error = 2
        cumulative_runtime = 0
        for epoch in range(num_iterations):
            # Randomly sample a new configuration
            config = np.empty(len(hyperparam_ranges))
            for i in range(len(hyperparam_ranges)):
                lower_bound, upper_bound = hyperparam_ranges[i]
                config[i] = np.random.randint(lower_bound, upper_bound + 1)
            # Update incumbent if new configuration performs better
            error = objective_function(config)
            if error < incumbent_error:
                incumbent = config
                incumbent_error = error
            incumbent_errors[epoch] += incumbent_error
            # Measure the runtime
            time = runtime(config)
            cumulative_runtime += time
            cumulative_runtimes[epoch] += cumulative_runtime
    incumbent_errors /= num_runs
    cumulative_runtimes /= num_runs
    # Write average incumbent error per epoch into file with given filename
    if output_filename is not None:
        with open(output_filename, "w") as output_file:
            for i in range(len(incumbent_errors)):
                output_file.write("%d\t%f\t%f\n" %
                        (i, incumbent_errors[i], cumulative_runtimes[i]))


def bayesian_optimization(num_iterations=50, num_runs=10, output_filename=None):
    """
    Run a Bayesian optimization procedure with the given number of iterations
    to find a good hyperparameter configuration on the network surrogate.
    Run the whole search a given number of times and write the average
    optimization trajectory (incumbent errors per epoch) as well as the
    cumulative runtime of all configuration evaluations per epoch into
    the file specified by the given filename.
    """
    lower_bounds = np.array([b[0] for b in hyperparam_ranges])
    upper_bounds = np.array([b[1] for b in hyperparam_ranges])
    incumbent_errors = np.zeros(num_iterations)
    cumulative_runtimes = np.zeros(num_iterations)
    for run in range(num_runs):
        results = fmin.bayesian_optimization(
                objective_function, lower_bounds, upper_bounds,
                num_iterations=num_iterations)
        cumulative_runtimes += \
                np.cumsum([runtime(config) for config in results["X"]])
        incumbent_errors += np.array(results["incumbent_values"])
        print("Finished run %d of %d." % (run, num_runs))
    incumbent_errors /= num_runs
    cumulative_runtimes /= num_runs
    # Write average incumbent error per epoch into file with given filename
    if output_filename is not None:
        with open(output_filename, "w") as output_file:
            for i in range(len(incumbent_errors)):
                output_file.write("%d\t%f\t%f\n" %
                        (i, incumbent_errors[i], cumulative_runtimes[i]))

if __name__ == "__main__":
    # random_search(output_filename="random-search-trajectory.txt")
    bayesian_optimization(output_filename="bayesian-optimization-trajectory.txt")

