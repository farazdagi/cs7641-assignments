import numpy as np

import experiments
import learners


class DTExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # TODO: Clean up the older alpha stuff?
        max_depths = np.arange(1, 51, 1)
        params = {"DT__criterion": ["gini", "entropy"], "DT__max_depth": max_depths,
                  "DT__class_weight": ["balanced", None]}  # , "DT__max_leaf_nodes": max_leaf_nodes}
        complexity_param = {"name": "DT__max_depth", "display_name": "Max Depth", "values": max_depths}

        # max_leaf_nodes = np.arange(10, 200, 10)
        # params = {"DT__criterion": ["gini", "entropy"],
        #           "DT__class_weight": ["balanced", None], "DT__max_leaf_nodes": max_leaf_nodes}
        # complexity_param = {
        #     "name": "DT__max_leaf_nodes", "display_name": "Max Leaf Nodes", "values": max_leaf_nodes}

        best_params = None
        # Uncomment to select known best params from grid search. This will skip the grid search and just rebuild
        # the various graphs
        #
        # Dataset 1:
        # Seed: 2702306879, 3882803657
        # best_params = {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 11}
        #
        # Dataset 2:
        # best_params = {"criterion": "entropy", "max_depth": 4, "class_weight": "balanced"}

        learner = learners.DTLearner(random_state=self._details.seed)
        if best_params is not None:
            learner.set_params(**best_params)
            self.log("Best parameters are provided, GridSearchCV will is skipped")
        else:
            self.log("Best parameters are not provided, GridSearchCV is scheduled")

        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name,
                                       learner, "DT", "DT", params,
                                       complexity_param=complexity_param, seed=self._details.seed,
                                       threads=self._details.threads,
                                       best_params=best_params,
                                       verbose=self._verbose)
