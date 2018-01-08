import sys
import numpy as np
import pickle
import argparse
import ConfigSpace as CS
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.ERROR)
from copy import deepcopy

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker


def create_config_space():
    cs = CS.ConfigurationSpace()

    params = {
        "Adam_final_lr_fraction":
            CS.UniformFloatHyperparameter("Adam_final_lr_fraction",
            lower=10e-4, upper=1.0, default_value=0.01, log=True),
        "Adam_initial_lr":
            CS.UniformFloatHyperparameter("Adam_initial_lr",
                lower=10e-4, upper=0.01, default_value=10e-3, log=True),
        "SGD_final_lr_fraction":
            CS.UniformFloatHyperparameter("SGD_final_lr_fraction",
                lower=10e-4, upper=1.0, default_value=0.01, log=True),
        "SGD_initial_lr":
            CS.UniformFloatHyperparameter("SGD_initial_lr",
                lower=10e-3, upper=0.5, default_value=0.1, log=True),
        "SGD_momentum":
            CS.UniformFloatHyperparameter("SGD_momentum",
                lower=0.0, upper=0.99, default_value=0.9),
        "StepDecay_epochs_per_step":
            CS.UniformIntegerHyperparameter("StepDecay_epochs_per_step",
                lower=1, upper=128, default_value=16, log=True),
        "activation":
            CS.CategoricalHyperparameter("activation",
                ["relu", "tanh"], default_value="relu"),
        "batch_size":
            CS.UniformIntegerHyperparameter("batch_size",
                lower=8, upper=256, default_value=16, log=True),
        "dropout_0":
            CS.UniformFloatHyperparameter("dropout_0",
                lower=0.0, upper=0.5, default_value=0.0),
        "dropout_1":
            CS.UniformFloatHyperparameter("dropout_1",
                lower=0.0, upper=0.5, default_value=0.0),
        "dropout_2":
            CS.UniformFloatHyperparameter("dropout_2",
                lower=0.0, upper=0.5, default_value=0.0),
        "dropout_3":
            CS.UniformFloatHyperparameter("dropout_3",
                lower=0.0, upper=0.5, default_value=0.0),
        "l2_reg_0":
            CS.UniformFloatHyperparameter("l2_reg_0",
                lower=10e-6, upper=0.01, default_value=10e-4, log=True),
        "l2_reg_1":
            CS.UniformFloatHyperparameter("l2_reg_1",
                lower=10e-6, upper=0.01, default_value=10e-4, log=True),
        "l2_reg_2":
            CS.UniformFloatHyperparameter("l2_reg_2",
                lower=10e-6, upper=0.01, default_value=10e-4, log=True),
        "l2_reg_3":
            CS.UniformFloatHyperparameter("l2_reg_3",
                lower=10e-6, upper=0.01, default_value=10e-4, log=True),
        "learning_rate_schedule":
            CS.CategoricalHyperparameter("learning_rate_schedule",
                ["ExponentialDecay", "StepDecay"],
                default_value="ExponentialDecay"),
        "loss_function":
            CS.CategoricalHyperparameter("loss_function",
                ["categorical_crossentropy"],
                default_value="categorical_crossentropy"),
        "num_layers":
            CS.UniformIntegerHyperparameter("num_layers",
                lower=1, upper=4, default_value=2),
        "num_units_0":
            CS.UniformIntegerHyperparameter("num_units_0",
                lower=16, upper=256, default_value=32, log=True),
        "num_units_1":
            CS.UniformIntegerHyperparameter("num_units_1",
                lower=16, upper=256, default_value=32, log=True),
        "num_units_2":
            CS.UniformIntegerHyperparameter("num_units_2",
                lower=16, upper=256, default_value=32, log=True),
        "num_units_3":
            CS.UniformIntegerHyperparameter("num_units_3",
                lower=16, upper=256, default_value=32, log=True),
        "optimizer":
            CS.CategoricalHyperparameter("optimizer",
                ["Adam", "SGD"], default_value="Adam"),
        "output_activation":
            CS.CategoricalHyperparameter("output_activation",
                ["softmax"], default_value="softmax")
    }
    conditions = [
        CS.EqualsCondition(
            params["Adam_final_lr_fraction"], params["optimizer"], "Adam"),
        CS.EqualsCondition(
            params["Adam_initial_lr"], params["optimizer"], "Adam"),
        CS.EqualsCondition(
            params["SGD_final_lr_fraction"], params["optimizer"], "SGD"),
        CS.EqualsCondition(
            params["SGD_initial_lr"], params["optimizer"], "SGD"),
        CS.EqualsCondition(
            params["SGD_momentum"], params["optimizer"], "SGD"),
        CS.EqualsCondition(
            params["StepDecay_epochs_per_step"],
            params["learning_rate_schedule"], "StepDecay"),
        CS.GreaterThanCondition(
            params["dropout_1"], params["num_layers"], 2),
        CS.GreaterThanCondition(
            params["dropout_2"], params["num_layers"], 3),
        CS.EqualsCondition(
            params["dropout_3"], params["num_layers"], 4),
        CS.GreaterThanCondition(
            params["l2_reg_1"], params["num_layers"], 2),
        CS.GreaterThanCondition(
            params["l2_reg_2"], params["num_layers"], 3),
        CS.EqualsCondition(
            params["l2_reg_3"], params["num_layers"], 4),
        CS.GreaterThanCondition(
            params["num_units_1"], params["num_layers"], 2),
        CS.GreaterThanCondition(
            params["num_units_2"], params["num_layers"], 3),
        CS.EqualsCondition(
            params["num_units_3"], params["num_layers"], 4)
    ]
    for param_name in params:
        cs.add_hyperparameter(params[param_name])
    for condition in conditions:
        cs.add_condition(condition)
    return cs


def objective_function(config, epoch=127, **kwargs):
    # Cast the config to an array such that it can be forwarded to the surrogate
    x = deepcopy(config.get_array())
    x[np.isnan(x)] = -1
    lc = rf.predict(x[None, :])[0]
    c = cost_rf.predict(x[None, :])[0]

    return lc[epoch], {"cost": c, "learning_curve": lc[:epoch].tolist()}


class WorkerWrapper(Worker):
    def compute(self, config, budget, *args, **kwargs):
        cfg = CS.Configuration(cs, values=config)
        loss, info = objective_function(cfg, epoch=int(budget))

        return ({
            'loss': loss,
            'info': {"runtime": info["cost"],
                     "lc": info["learning_curve"]}
        })


def plot_incumbent_curve(wallclock_times, incumbent_errors, filename):
    """ Plot incumbent validation error and wallclock time for each iteration.
    """
    fig, ax = plt.subplots()
    ax.plot(wallclock_times, incumbent_errors, marker=".")
    ax.set(xlabel="Wallclock time", ylabel="Incumbent Validation Error")
    fig.savefig(filename)


def plot_all_learning_curves(learning_curves, filename):
    """ Plot learning curves for all evaluated configurations.
    """
    fig, ax = plt.subplots()
    for learning_curve in learning_curves:
        ax.plot(learning_curve)
    ax.set(xlabel="Epoch", ylabel="Validation Error")
    fig.savefig(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_smac', action='store_true')
    parser.add_argument('--run_hyperband', action='store_true')
    parser.add_argument('--n_iters', default=50, type=int)
    args = vars(parser.parse_args())

    n_iters = args['n_iters']

    cs = create_config_space()
    rf = pickle.load(open("./rf_surrogate_paramnet_mnist.pkl", "rb"))
    cost_rf = pickle.load(open("./rf_cost_surrogate_paramnet_mnist.pkl", "rb"))

    if args["run_smac"]:
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": n_iters,
                             "cs": cs,
                             "deterministic": "true",
                             "output_dir": ""})

        smac = SMAC(scenario=scenario, tae_runner=objective_function)
        smac.optimize()

        # The following lines extract the incumbent strategy and the estimated
        # wall-clock time of the optimization
        rh = smac.runhistory
        incumbents = []
        incumbent_performance = []
        inc = None
        inc_value = 1
        idx = 1
        t = smac.get_trajectory()

        wall_clock_time = []
        cum_time = 0
        for d in rh.data:
            cum_time += rh.data[d].additional_info["cost"]
            wall_clock_time.append(cum_time)
        for i in range(n_iters):

            if idx < len(t) and i == t[idx].ta_runs - 1:
                inc = t[idx].incumbent
                inc_value = t[idx].train_perf
                idx += 1

            incumbents.append(inc)
            incumbent_performance.append(inc_value)

        lc_smac = []
        for d in rh.data:
            lc_smac.append(rh.data[d].additional_info["learning_curve"])

        # Plot curves
        plot_incumbent_curve(wall_clock_time, incumbent_performance,
                             "smac-incumbent-curve.png")
        plot_all_learning_curves(lc_smac, "smac-all-learning-curves.png")

    if args["run_hyperband"]:
        nameserver, ns_port = \
                hpbandster.distributed.utils.start_local_nameserver()

        # starting the worker in a separate thread
        w = WorkerWrapper(nameserver=nameserver, ns_port=ns_port)
        w.run(background=True)

        CG = hpbandster.config_generators.RandomSampling(cs)

        # instantiating Hyperband with some minimal configuration
        HB = hpbandster.HB_master.HpBandSter(
            config_generator=CG,
            run_id='0',
            eta=2,  # defines downsampling rate
            min_budget=1,  # minimum number of epochs / minimum budget
            max_budget=127,  # maximum number of epochs / maximum budget
            nameserver=nameserver,
            ns_port=ns_port,
            job_queue_sizes=(0, 1),
        )
        # runs one iteration if at least one worker is available
        res = HB.run(10, min_n_workers=1)

        # shutdown the worker and the dispatcher
        HB.shutdown(shutdown_workers=True)

        # extract incumbent trajectory and all evaluated learning curves
        traj = res.get_incumbent_trajectory()
        wall_clock_time = []
        cum_time = 0

        for c in traj["config_ids"]:
            cum_time += res.get_runs_by_id(c)[-1]["info"]["runtime"]
            wall_clock_time.append(cum_time)

        lc_hyperband = []
        for r in res.get_all_runs():
            c = r["config_id"]
            lc_hyperband.append(res.get_runs_by_id(c)[-1]["info"]["lc"])

        incumbent_performance = traj["losses"]

        # Plot curves
        plot_incumbent_curve(wall_clock_time, incumbent_performance,
                             "hpband-incumbent-curve.png")
        plot_all_learning_curves(lc_hyperband, "hpband-all-learning-curves.png")

