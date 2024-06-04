from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.utils import resample
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
import numpy as np
import pandas as pd
import scipy.stats
import warnings
from multiprocessing import Pool
from functools import partial


def get_error_metrics(y_true_values: np.array, y_predicted_values: np.array) -> dict[str, float]:
    """Computes a number of error metrics, returned as a dictionary containing the error metrics names and values.
    Args:
        y_true_values (np.array): The true classes for a dataset
        y_predicted_values (np.array): The model predicted classes for the same dataset as y_true_values
    Returns:
        dict[str, float]: A dictionary of metric name to metric value
    """

    metrics = dict()
    metrics["accuracy"] = accuracy_score(y_true_values, y_predicted_values)
    metrics["recall positive class"] = recall_score(
        y_true_values, y_predicted_values, pos_label=1, zero_division=0.0)
    metrics["recall negative class"] = recall_score(
        y_true_values, y_predicted_values, pos_label=0, zero_division=0.0)
    metrics["recall macro average"] = recall_score(
        y_true_values, y_predicted_values, average="macro", zero_division=0.0)
    metrics["precision positive class"] = precision_score(
        y_true_values, y_predicted_values, pos_label=1, zero_division=0.0)
    metrics["precision negative class"] = precision_score(
        y_true_values, y_predicted_values, pos_label=0, zero_division=0.0)
    metrics["precision macro average"] = precision_score(
        y_true_values, y_predicted_values, average="macro", zero_division=0.0)
    metrics["f1 score positive class"] = f1_score(
        y_true_values, y_predicted_values, pos_label=1, zero_division=0.0)
    metrics["f1 score negative class"] = f1_score(
        y_true_values, y_predicted_values, pos_label=0, zero_division=0.0)
    metrics["f1 score macro average"] = f1_score(
        y_true_values, y_predicted_values, average="macro", zero_division=0.0)
    metrics["Matthews correlation coefficient"] = abs(
        matthews_corrcoef(y_true_values, y_predicted_values))

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

    df_true_class = pd.DataFrame({
        "Is Privileged": privilege_status,
        "Class Label": y_true_values
    })

    df_predicted_class = pd.DataFrame({
        "Is Privileged": privilege_status,
        "Class Label": y_predicted_values
    })

    # I don't believe the values of protected_attribute_names and privileged_classes matter here since this is really set in ClassificationMetric, however these are required fields so might as well set them reasonably.
    ds_true_class = StandardDataset(
        df_true_class,
        label_name="Class Label",
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"],
        privileged_classes=[[1]]
    )

    ds_predicted_class = StandardDataset(
        df_predicted_class,
        label_name="Class Label",
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"],
        privileged_classes=[[1]]
    )

    classification_metric = ClassificationMetric(ds_true_class, ds_predicted_class, unprivileged_groups=[
                                                 {"Is Privileged": False}], privileged_groups=[{"Is Privileged": True}])

    metrics = dict()

    # TODO These seem to raise a RuntimeWarning: invalid value encountered in double_scalars often, which makes reading the output annoying
    # I believe this is because these methods calculate some values they don't actually require, but I should find a way to double check this
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        metrics["statistical parity difference"] = abs(
            classification_metric.statistical_parity_difference())
        metrics["average odds difference"] = classification_metric.average_abs_odds_difference()
        metrics["equal opportunity difference"] = abs(
            classification_metric.equal_opportunity_difference())
        metrics["false positive rate difference"] = abs(
            classification_metric.false_positive_rate_difference())
        metrics["error rate difference"] = abs(
            classification_metric.error_rate_difference())

    return metrics


def get_all_metrics(y_true_values: np.array, y_predicted_values: np.array, privilege_status: np.array) -> dict[str, dict[str, float]]:
    """Computes a number of fairness and error metrics, returned as a dictionary containing the metric type, followed by the metric names and values.
    Args:
        y_true_values (np.array): The true classes for a dataset
        y_predicted_values (np.array): The model predicted classes for the same dataset as y_true_values
        privilege_status (np.array): Whether the individual to which each class is assigned belongs to the privileged group or unprivileged group
    Returns:
        dict[str, dict[str, float]]: A dictionary of metric type to a dictionary of metric name to metric value

    For example, the metric type of "error rate difference" is fairness so it would be accessed as result["fairness"]["error rate difference"].  A more detailed structure of the result is shown below.
    result = {
        "fairness" : {
            "error rate difference" : 0.38,
            ...
        },
        "error" : {...}   
    }
    """

    return {
        "fairness": get_fairness_metrics(y_true_values, y_predicted_values, privilege_status),
        "error": get_error_metrics(y_true_values, y_predicted_values)
    }


def _bootstrap_sample_metrics(y_true_values: np.array, y_predicted_values: np.array, privilege_status: np.array, random_state: int=None, stratify: bool=False):
    """Helper function for bootstrap_all_metrics to allow multiprocessing. Computes a single bootstrap sample and returns the required metrics for it. y_true_values, y_predicted_values, privilege_status, and stratify mirror the arguments of bootstrap_all_metrics, while random_state is expected to be an integer 
    """
    return get_all_metrics(
        *resample(
            y_true_values,
            y_predicted_values,
            privilege_status,
            random_state=random_state,
            stratify=privilege_status if stratify else None
        )
    )


def bootstrap_all_metrics(y_true_values: np.array, y_predicted_values: np.array, privilege_status: np.array, repetitions: int=200, stratify: bool=False, seed: int=None) -> dict[str, dict[str, dict[str, float]]]:
    """Applies bootstrap resampling in order to estimate a number of summary statistics on the results found by get_all_metrics
    Args:
        y_true_values (np.array): The true classes for a dataset
        y_predicted_values (np.array): The model predicted classes for the same dataset as y_true_values
        privilege_status (np.array): Whether the individual to which each class is assigned belongs to the privileged group or unprivileged group
        repetitions (int, optional): How many bootstrap samples should be taken
        stratify (bool, optional): Whether to stratify the bootstrap samples by privilege_status, ensuring representation of both groups match that of the original sample
        seed (int, optional): Any valid input to np.random.default_rng, used for repeatability of a result
    Returns:
        dict[str, dict[str, dict[str, float]]]: A dictionary of metric type to a dictionary of metric name to a dictionary of summary statistic name to value

    For example, the metric type of "error rate difference" is fairness, and we may be interested in its standard deviation. This would be accessed as result["fairness"]["error rate difference"]["standard deviation"]. A more detailed structure of the result is shown below.
    result = {
        "fairness" : {
            "error rate difference" : {
                "standard deviation" : 0.24,
                ...
            },
            ...
        },
        "error" : {...}   
    }
    """

    # We expect that these will all be boolean arrays, so make sure they are of this type to reduce memory usage
    y_true_values = y_true_values.astype(bool)
    y_predicted_values = y_predicted_values.astype(bool)
    privilege_status = privilege_status.astype(bool)

    base_values = get_all_metrics(
        y_true_values, y_predicted_values, privilege_status)

    # Since sklearn.utils.resample does not support using np.random.default_rng directly, we will generate random numbers
    # to use as the random state argument. We set the range of these randomly generated numbers to the square of the number
    # of repetitions since this gives us around a 0.6 probability of having no repeats. A couple of repeats is ok, but
    # we want the samples to be as independent as possible. A cube would be better, giving us around a 0.9 probability but
    # would restrict repetitions from exceeding 1000 by much. 
    # We take the random values approach to increase independence between runs with different seeds
    # TODO an alternative would be to take seed * some_number * i, but leaves open some highly correlated seed values
    rng = np.random.default_rng(seed)
    random_range = repetitions ** 2

    # Apply multiprocessing to save time, the order in which we get the statistics doesn't matter
    bootstrap_partial = partial(_bootstrap_sample_metrics, y_true_values,
                                y_predicted_values, privilege_status, stratify=stratify)
    with Pool(processes=4) as pool:
        bootstrapped_metrics = list(
            pool.imap_unordered(
                bootstrap_partial,
                [rng.integers(random_range) for _ in range(repetitions)],
                chunksize=50
            )
        )

    # Create the dictionary to return containing the calculated statistics about each metric
    metric_stats = dict()
    for metric_type in bootstrapped_metrics[0].keys():
        metric_stats[metric_type] = dict()

        for metric_name in bootstrapped_metrics[0][metric_type].keys():
            metric_results = [result[metric_type][metric_name]
                              for result in bootstrapped_metrics]

            # For some metrics and subsamples we get NaN results, which makes all our statistics NaN if included. Thus, exclude any NaN values
            metric_results = [
                result for result in metric_results if not np.isnan(result)]
            if len(metric_results) == 0:
                continue

            value = base_values[metric_type][metric_name]
            min_value, lower_confidence_interval, lower_quartile, median, upper_quartile, upper_confidence_interval, max_value = np.percentile(
                metric_results, [0, 17, 25, 50, 75, 83, 100])

            metric_stats[metric_type][metric_name] = {
                "value": value,
                "mean": np.mean(metric_results),
                "median": median,
                "confidence interval": [lower_confidence_interval, upper_confidence_interval],
                "inter quartile range": [lower_quartile, upper_quartile],
                "range": [min_value, max_value],
                "standard deviation": np.std(metric_results),
                "skew": scipy.stats.skew(metric_results),
                "kurtosis": scipy.stats.kurtosis(metric_results)
            }

    return metric_stats
