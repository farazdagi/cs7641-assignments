import argparse
import logging
import numpy as np
from datetime import datetime

import experiments
from data import loader

logger = logging.getLogger(__name__)

DATASETS = {
    "spam": {
        "loader": loader.SpamData,
        "readable_name": "Spam",
    },
    "credit_default": {
        "loader": loader.CreditDefaultData,
        "readable_name": "Credit Default",
    },
    "pen_digits": {
        "loader": loader.PenDigitData,
        "readable_name": "Handwritten Digits",
    },
    "abalone": {
        "loader": loader.AbaloneData,
        "readable_name": "Abalone",
    },
    "htru": {
        "loader": loader.HTRU2Data,
        "readable_name": "HTRU2",
    },
}


def run_experiment(experiment_details, experiment, timing_key, verbose, timings):
    exp = experiment(experiment_details, verbose=verbose)
    exp.set_logger(logger)

    logger.info("Experiment started: {} on {} dataset".format(timing_key, experiment_details.ds_readable_name))
    t = datetime.now()
    exp.perform()

    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform some SL experiments")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads (defaults to 1, -1 for auto)")
    parser.add_argument("--seed", type=int, help="A random seed to set, if desired")
    parser.add_argument("--ann", action="store_true", help="Run the ANN experiment")
    parser.add_argument("--boosting", action="store_true", help="Run the Boosting experiment")
    parser.add_argument("--dt", action="store_true", help="Run the Decision Tree experiment")
    parser.add_argument("--knn", action="store_true", help="Run the KNN experiment")
    parser.add_argument("--svm", action="store_true", help="Run the SVM experiment")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--dataset", default="spam", help="Dataset name", choices=DATASETS.keys())
    parser.add_argument("--verbose", action="store_true", help="If true, provide verbose output")
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads
    dataset = DATASETS[args.dataset]

    # select random seed
    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
    logger.info("Seed: {}".format(seed))

    logger.info("Available datasets: {}".format(DATASETS.keys()))
    logger.info("Selected dataset: {}".format(args.dataset))

    timings = {}
    data_loader = dataset["loader"](verbose=verbose, seed=seed)
    data_loader.set_logger(logger)
    data_loader.load_and_process()
    data_loader.build_train_test_split()
    data_loader.scale_standard()
    experiment_details = experiments.ExperimentDetails(
        data_loader, args.dataset, dataset["readable_name"], threads=threads, seed=seed
    )

    if args.ann or args.all:
        run_experiment(experiment_details, experiments.ANNExperiment, "ANN", verbose, timings)

    if args.boosting or args.all:
        run_experiment(experiment_details, experiments.BoostingExperiment, "Boosting", verbose, timings)

    if args.dt or args.all:
        run_experiment(experiment_details, experiments.DTExperiment, "DT", verbose, timings)

    if args.knn or args.all:
        run_experiment(experiment_details, experiments.KNNExperiment, "KNN", verbose, timings)

    if args.svm or args.all:
        run_experiment(experiment_details, experiments.SVMExperiment, "SVM", verbose, timings)

    print(timings)
