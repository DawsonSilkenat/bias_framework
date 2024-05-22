import pandas as pd
import numpy as np
from .metrics import bootstrap_all_metrics
from sklearn.calibration import CalibratedClassifierCV
from .debiasing_graphs import DebiasingGraphsObject
from .baselines.fairea_curve import FaireaCurve
from .dataset_processing import covert_to_datasets_train, covert_to_datasets_validation
from .debiasing import no_debiasing, learning_fair_representation, reweighting, reject_option_classification, calibrated_equal_odds, equal_odds

# This is only for information on runtime rather than used functionally
import time

# aif360 seems to do something pandas doesn't like, and it makes it hard to debug when all I read are these warning messages
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class Bias_Framework:
    def __init__(self, model, df_x_train: pd.DataFrame, df_x_validation: pd.DataFrame, df_y_train: pd.DataFrame, df_y_validation: pd.DataFrame, pre_processing=None) -> None:
        """Creates an instance of the bias framework applied to the specified model and data

        Args:
            model: The ML model to which the bias framework will be applied. This model must have a fit method and predict method. I am assuming at the moment that this will be an sklearn ml model, might try to modify this to be more flexible later.
            df_x_train (pd.DataFrame): The training data features for the ML model.
            df_x_validation (pd.DataFrame): The validation data features for the ML model.
            df_y_train (pd.DataFrame): The training data labels for the ML model.
            df_y_validation (pd.DataFrame): The validation data labels for the ML model.
            pre_processing (optional): Any pre-processing to apply to the data before use by the ml model. Expected to implement methods for fit_transform and transform.
        """

        # Not all models implement a function for prediction probability. Probabilities are required for some postprocessing debiasing. The CalibratedClassifierCV method preforms probability calibration, which also adds this functionality if it is not part of the model
        # TODO create class which defines the required interface, remove this line of code and place responsibility on user to satisfy interface
        self.model = CalibratedClassifierCV(estimator=model)

        self.df_x_train = df_x_train.copy()
        self.df_x_validation = df_x_validation.copy()

        self.y_train = df_y_train.copy().to_numpy().ravel()
        self.y_validation = df_y_validation.copy().to_numpy().ravel()

        self.privilege_train = np.ones(len(df_x_train))
        self.privilege_validation = np.ones(len(df_y_train))

        self.pre_processing = pre_processing

        self.__debiasing_graph = None
        self.__metrics_by_debiasing_technique = dict()
        self.__raw_results = dict()

    def set_privilege_function(self, privilege_function: callable) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class

        Args:
            privilege_function (callable): A function which when applied to a row of a dataframe will return a 1 for privileged and 0 for unprivileged. True and false values should also work.
        """

        self.privilege_train = self.df_x_train.apply(
            privilege_function, axis=1).to_numpy()
        self.privilege_validation = self.df_x_validation.apply(
            privilege_function, axis=1).to_numpy()

    def set_privileged_combinations(self, privileged_combinations: list[dict[str: any]]) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class by specifying which groupings are considered privileged

        Args:
            privileged_combinations: a list of dictionaries, each dictionary maps a set of columns name to the values they must take. Note: Each columns name must map to a single value, not a list of possible values. The values of any column names not included in the dictionary are ignored

            Example from aif360 documentation:
            [{'sex': 1, 'age': 1}, {'sex': 0}]
            The first dictionary indicates that if sex has value 1 and age has value 1 then the individual belongs to the privileged group
            The second dictionary indicates that if sex has value 0 then the individual belongs to the privileged group (regardless of age)
        """
        def privilege_function(x): return int(
            # Checks if a row matches any of the valid groupings
            any([
                # Checks if all the requirements for a particular grouping are met
                all([x[name] == value for name, value in grouping]) for grouping in privileged_combinations])
        )
        self.set_privilege_function(privilege_function)

    def set_unprivileged_combinations(self, unprivileged_combinations: list[dict[str: any]]) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class by specifying which groupings are not considered privileged

        Args:
            unprivileged_combinations: a list of dictionaries, each dictionary maps a set of columns name to the values they must take. Note: Each columns name must map to a single value, not a list of possible values. The values of any column names not included in the dictionary are ignored

            Example from aif360 documentation:
            [{'sex': 1, 'age': 1}, {'sex': 0}]
            The first dictionary indicates that if sex has value 1 and age has value 1 then the individual belongs to the unprivileged group
            The second dictionary indicates that if sex has value 0 then the individual belongs to the unprivileged group (regardless of age)
        """
        def privilege_function(x): return int(
            # Checks if a row matches any of the valid groupings, and negates it so we are unprivileged
            not any([
                # Checks if all the requirements for a particular grouping are met
                all([x[name] == value for name, value in grouping]) for grouping in unprivileged_combinations])
        )
        self.set_privilege_function(privilege_function)

    def run_framework(self, seed=None):
        x_train = self.df_x_train
        x_validation = self.df_x_validation
        if self.pre_processing:
            x_train = self.pre_processing.fit_transform(x_train, self.y_train)
            x_validation = self.pre_processing.transform(x_validation)

        train_predictions, training_probabilities, validation_predictions, validation_probabilities = no_debiasing(
            self.model, x_train, x_validation, self.y_train)
        
        # TODO remove this
        self.probabilities = [training_probabilities, validation_probabilities]
        

        start = time.time()
        fairea_curve = FaireaCurve(
            self.y_validation, validation_predictions, self.privilege_validation)
        print(f"{time.time() - start} seconds to get fairea baseline")

        ds_train_true_labels, ds_train_predictions = covert_to_datasets_train(
            x_train, self.y_train, train_predictions, training_probabilities, self.privilege_train)
        ds_validation_predictions, ds_validation_no_labels = covert_to_datasets_validation(
            x_validation, validation_predictions, validation_probabilities, self.privilege_validation)
        
        
        self.probabilities.append(np.copy(ds_train_predictions.scores))
        self.probabilities.append(np.copy(ds_validation_predictions.scores))

        raw_results = dict()

        start = time.time()
        debiasing_result = learning_fair_representation(
            self.model, ds_train_true_labels, ds_validation_no_labels, seed=seed)
        raw_results.update(debiasing_result)
        print(f"{time.time() - start} seconds to run learning fair representation")

        start = time.time()
        debiasing_result = reweighting(
            self.model, ds_train_true_labels, ds_validation_no_labels)
        raw_results.update(debiasing_result)
        print(f"{time.time() - start} seconds to run reweighting")
        
        self.probabilities.append(ds_train_predictions.scores)
        self.probabilities.append(ds_validation_predictions.scores)

        start = time.time()
        debiasing_result = reject_option_classification(
            ds_train_true_labels, ds_train_predictions, ds_validation_predictions)
        raw_results.update(debiasing_result)
        print(f"{time.time() - start} seconds to run reject option classification")
        
        self.probabilities.append(np.copy(ds_train_predictions.scores))
        self.probabilities.append(np.copy(ds_validation_predictions.scores))

        start = time.time()
        debiasing_result = calibrated_equal_odds(
            ds_train_true_labels, ds_train_predictions, ds_validation_predictions, seed=seed)
        raw_results.update(debiasing_result)
        print(f"{time.time() - start} seconds to run calibrated equal odds")

        self.probabilities.append(np.copy(ds_train_predictions.scores))
        self.probabilities.append(np.copy(ds_validation_predictions.scores))

        start = time.time()
        debiasing_result = equal_odds(
            ds_train_true_labels, ds_train_predictions, ds_validation_predictions, seed=seed)
        raw_results.update(debiasing_result)
        print(f"{time.time() - start} seconds to run equal odds")
        
        self.probabilities.append(np.copy(ds_train_predictions.scores))
        self.probabilities.append(np.copy(ds_validation_predictions.scores))

        self.__metrics_by_debiasing_technique = {key: bootstrap_all_metrics(
            self.y_validation, value, self.privilege_validation, seed=seed) for key, value in raw_results.items()}
        self.__debiasing_graph = DebiasingGraphsObject(
            self.__metrics_by_debiasing_technique, fairea_curve)

    def get_debias_methodologies(self) -> list[str]:
        if self.__debiasing_graph is None:
            return []
        return self.__debiasing_graph.get_debias_methodologies()

    def get_error_metric_names(self) -> list[str]:
        if self.__debiasing_graph is None:
            return []
        return self.__debiasing_graph.get_error_metric_names()

    def get_bias_metric_names(self) -> list[str]:
        if self.__debiasing_graph is None:
            return []
        return self.__debiasing_graph.get_bias_metric_names()

    def get_raw_data(self) -> dict[str, dict[str, dict[str, float]]]:
        if self.__debiasing_graph is None:
            return dict()
        return self.__debiasing_graph.get_raw_data()

    def get_DebiasingGraphsObject(self) -> DebiasingGraphsObject:
        return self.__debiasing_graph

    def show_single_graph(self, error_metric: str, fairness_metric: str) -> None:
        self.__debiasing_graph.show_single_graph(error_metric, fairness_metric)

    def show_subplots(self, error_metrics: list[str], fairness_metrics: list[str]) -> None:
        self.__debiasing_graph.show_subplots(error_metrics, fairness_metrics)

    def show_all_subplots(self) -> None:
        self.__debiasing_graph.show_all_subplots()
