from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.utils import resample
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
import numpy as np
import pandas as pd
import scipy.stats


def get_error_metrics(y_true_values: np.array, y_predicted_values: np.array) -> dict[str, float]:
    """Computes a number of error metrics, returned as a dictionary containing the error metrics names and values.
    Args:
        y_true_values (np.array): The true classes for a dataset
        y_predicted_values (np.array): The model predicted classes for the same dataset as y_true_values
    Returns:
        dict[str, float]: A dictionary of metric name to metric value
    """

    # These metrics were selected due to their use in the Bias Mitigation Methods paper.
    # There exist more we could include, but not a priority at this stage
    metrics = dict()
    metrics["accuracy"]                         = accuracy_score(y_true_values, y_predicted_values)
    metrics["recall positive class"]            = recall_score(y_true_values, y_predicted_values, pos_label=1, zero_division=0.0)
    metrics["recall negative class"]            = recall_score(y_true_values, y_predicted_values, pos_label=0, zero_division=0.0)
    metrics["recall macro average"]             = recall_score(y_true_values, y_predicted_values, average="macro", zero_division=0.0)
    metrics["precision positive class"]         = precision_score(y_true_values, y_predicted_values, pos_label=1, zero_division=0.0)
    metrics["precision negative class"]         = precision_score(y_true_values, y_predicted_values, pos_label=0, zero_division=0.0)
    metrics["precision macro average"]          = precision_score(y_true_values, y_predicted_values, average="macro", zero_division=0.0)
    metrics["f1 score positive class"]          = f1_score(y_true_values, y_predicted_values, pos_label=1, zero_division=0.0)
    metrics["f1 score negative class"]          = f1_score(y_true_values, y_predicted_values, pos_label=0, zero_division=0.0)
    metrics["f1 score macro average"]           = f1_score(y_true_values, y_predicted_values, average="macro", zero_division=0.0)
    metrics["Matthews correlation coefficient"] = abs(matthews_corrcoef(y_true_values, y_predicted_values))

    return metrics


def get_fairness_metrics(y_true_values: np.array, y_predicted_values: np.array, privilege_status: np.array) -> dict[str, float]:
    """Computes a number of fairness metrics, returned as a dictionary containing the fairness metrics names and values.
    Args:
        y_true_values (np.array): The true classes for a dataset
        y_predicted_values (np.array): The model predicted classes for the same dataset as y_true_values
        privilege_status (np.array): Whether the individual to which each class is assigned belongs to the privileged group or unprivileged group
    Returns:
        dict[str, float]: A dictionary of metric name to metric value
    """

    df_dataset_with_true_class = pd.DataFrame({
        "Is Privileged" : privilege_status,
        "Class Label" : y_true_values
    })

    df_dataset_with_predicted_class = pd.DataFrame({
        "Is Privileged" : privilege_status,
        "Class Label" : y_predicted_values
    })


    # I don't believe the values of protected_attribute_names and privileged_classes matter here since this is really
    # set in ClassificationMetric, however these are required fields so might as well set them reasonably.
    dataset_with_true_class = StandardDataset(
        df_dataset_with_true_class,
        label_name="Class Label",
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"],
        privileged_classes=[[1]]
    )

    dataset_with_predicted_class = StandardDataset(
        df_dataset_with_predicted_class,
        label_name="Class Label",
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"],
        privileged_classes=[[1]]
    )

    classification_metric = ClassificationMetric(dataset_with_true_class, dataset_with_predicted_class,
                                                 unprivileged_groups=[{"Is Privileged" : 0}],
                                                 privileged_groups=[{"Is Privileged" : 1}])

    metrics = dict()
    metrics["statistical parity difference"] = abs(classification_metric.statistical_parity_difference())
    metrics["average odds difference"]       = abs(classification_metric.average_abs_odds_difference())
    metrics["equal opportunity difference"]  = abs(classification_metric.equal_opportunity_difference())
    metrics["error rate difference"]         = abs(classification_metric.error_rate_difference())

    return metrics


def get_all_metrics(y_true_values: np.array, y_predicted_values: np.array, privilege_status: np.array) -> dict[str, dict[str, float]]:
    """Computes a number of fairness and error metrics, returned as a dictionary containing the metric type, followed by the metric names and values.
    Args:
        y_true_values (np.array): The true classes for a dataset
        y_predicted_values (np.array): The model predicted classes for the same dataset as y_true_values
        privilege_status (np.array): Whether the individual to which each class is assigned belongs to the privileged group or unprivileged group
    Returns:
        dict[str, dict[str, float]]: A dictionary of metric type to a dictionary of metric name to metric value
    """
    # TODO: add an example returned field here because this is confusing!

    return {
        "fairness" : get_fairness_metrics(y_true_values, y_predicted_values, privilege_status),
        "error" : get_error_metrics(y_true_values, y_predicted_values)
    }


def bootstrap_all_metrics(y_true_values: np.array, y_predicted_values: np.array, privilege_status: np.array,
                          repetitions: int = 100, seed=None) -> dict[str, dict[str, dict[str, float]]]:
    """Applies bootstrap resampling in order to estimate a number of summary statistics on the results found by get_all_metrics
    Args:
        y_true_values (np.array): The true classes for a dataset
        y_predicted_values (np.array): The model predicted classes for the same dataset as y_true_values
        privilege_status (np.array): Whether the individual to which each class is assigned belongs to the privileged group or unprivileged group
        repetitions (int, optional): How many bootstrap samples should be taken
        seed: Any valid input to np.random.default_rng, used for repeatability 
    Returns:
        dict[str, dict[str, dict[str, float]]]: A dictionary of metric type to a dictionary of metric name to a dictionary of summary statistic name to value
    """
    # TODO: add an example returned field here because this is confusing!

    rng = None
    if seed is not None:
        rng = np.random.default_rng(seed)

    # Record the result of each repetition 
    bootstrapped_metrics = []
    for _ in range(repetitions):
        true_values_boot, predicted_values_boot, privilege_status_boot = resample(y_true_values, y_predicted_values, privilege_status, random_state=rng)
        bootstrapped_metrics.append(get_all_metrics(true_values_boot, predicted_values_boot, privilege_status_boot))

    # Create the dictionary to return containing the calculated statistics about each metric
    metric_stats = dict()
    for metric_type in bootstrapped_metrics[0].keys():
        metric_stats[metric_type] = dict()

        for metric_name in bootstrapped_metrics[0][metric_type].keys():
            metric_results = [result[metric_type][metric_name]  for result in bootstrapped_metrics]
            value = np.mean(metric_results)
            std = np.std(metric_results)

            metric_stats[metric_type][metric_name] = {
                "value": value,
                "standard deviation": std,
                # Contains nan values if std == 0, but required for plotting 
                "confidence interval": scipy.stats.norm.interval(0.95, loc=value, scale=std) if std != 0 else (value, value),
                "skew": scipy.stats.skew(metric_results),
                "kurtosis": scipy.stats.kurtosis(metric_results),
                "quartiles": np.percentile(metric_results, [0, 25, 50, 75, 100])
            }

    return metric_stats


