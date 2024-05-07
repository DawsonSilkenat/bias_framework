from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.utils import resample
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
import numpy as np
import pandas as pd
import scipy.stats 


def get_error_metrics(y_true_value: np.array, y_predicted_value: np.array) -> dict[str, float]:
    """Computes a number of error metrics returns a dictionary containing the error metrics names and values.
    Args:
        y_true_value (np.array): The true classes for a dataset
        y_predicted_value (np.array): The model predicted classes for the same dataset as y_true_value
    Returns:
        dict[str, float]: A dictionary of metric name to metric value
    """

    # These metrics were selected due to their use in the Bias Mitigation Methods paper. There exist more we could include, but not a priority at this stage
    metrics = dict()
    metrics["accuracy"]                         = accuracy_score(y_true_value, y_predicted_value)
    metrics["recall positive class"]            = recall_score(y_true_value, y_predicted_value, pos_label=1, zero_division=0.0)
    metrics["recall negative class"]            = recall_score(y_true_value, y_predicted_value, pos_label=0, zero_division=0.0)
    metrics["recall macro average"]             = recall_score(y_true_value, y_predicted_value, average="macro", zero_division=0.0)
    metrics["precision positive class"]         = precision_score(y_true_value, y_predicted_value, pos_label=1, zero_division=0.0)
    metrics["precision negative class"]         = precision_score(y_true_value, y_predicted_value, pos_label=0, zero_division=0.0)
    metrics["precision macro average"]          = precision_score(y_true_value, y_predicted_value, average="macro", zero_division=0.0)
    metrics["f1 score positive class"]          = f1_score(y_true_value, y_predicted_value, pos_label=1, zero_division=0.0)
    metrics["f1 score negative class"]          = f1_score(y_true_value, y_predicted_value, pos_label=0, zero_division=0.0)
    metrics["f1 score macro average"]           = f1_score(y_true_value, y_predicted_value, average="macro", zero_division=0.0)
    metrics["Matthews correlation coefficient"] = abs(matthews_corrcoef(y_true_value, y_predicted_value))

    return metrics


def get_fairness_metrics(true_values: np.array, predicted_values: np.array, privilege_status: np.array) -> dict[str, float]:    
    
    
    df_dataset_with_true_class = pd.DataFrame({
        "Is Privileged" : privilege_status, 
        "Class Label" : true_values
    })
    
    df_dataset_with_predicted_class = pd.DataFrame({
        "Is Privileged" : privilege_status, 
        "Class Label" : predicted_values
    })
    
    
    # I don't believe the values of protected_attribute_names and privileged_classes matter here since this is really set in ClassificationMetric, however these are required fields so might as well set them reasonably. 
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
    

    classification_metric = ClassificationMetric(dataset_with_true_class, dataset_with_predicted_class, unprivileged_groups=[{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}])

    metrics = dict()
    metrics["statistical parity difference"] = abs(classification_metric.statistical_parity_difference())
    metrics["average odds difference"]       = abs(classification_metric.average_abs_odds_difference())
    metrics["equal opportunity difference"]  = abs(classification_metric.equal_opportunity_difference())
    metrics["error rate difference"]         = abs(classification_metric.error_rate_difference())
    
    return metrics


def get_all_metrics(true_values: np.array, predicted_values: np.array, privilege_status: np.array) -> dict[str, dict[str, float]]:
    return {
        "fairness" : get_fairness_metrics(true_values, predicted_values, privilege_status),
        "error" : get_error_metrics(true_values, predicted_values)
    }
    
    
def bootstrap_all_metrics(true_values: np.array, predicted_values: np.array, privilege_status: np.array, repetitions: int = 100, seed=None) -> dict[str, dict[str, dict[str, float]]]:
    
    rng = None
    # Set the random number generator for reproduceable results
    if seed is not None:
        rng = np.random.default_rng(seed)
    
    # Record the result of each repetition 
    bootstrapped_metrics = []
    for _ in range(repetitions):
        true_values_boot, predicted_values_boot, privilege_status_boot = resample(true_values, predicted_values, privilege_status, random_state=rng)
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
                # Contains nan values if std == 0, but required for ploting 
                "confidence interval": scipy.stats.norm.interval(0.95, loc=value, scale=std) if std != 0 else (value, value), 
                "skew": scipy.stats.skew(metric_results),
                "kurtosis": scipy.stats.kurtosis(metric_results),
                "quartiles": np.percentile(metric_results, [0, 25, 50, 75, 100]) 
            }
            
    return metric_stats   

  
def fairea_model_mutation(true_values: np.array, predicted_values: np.array, privilege_status: np.array, fractions_to_mutate: list[float] = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], repetitions=50) -> list[tuple[float, dict[str, dict[str, float]]]]:
    # Source: https://solar.cs.ucl.ac.uk/pdf/hort2021fairea.pdf
    
    # The Fairea paper used the mutation strategy where all values were mutated to the same label, and the label to mutate to was the one which produced the highest accuracy with 100% mutation. We follow this suggestion.
    mutate_to_value = np.argmax(np.bincount(true_values))
    
    metrics = []
    
    for mutation_fraction in fractions_to_mutate:
        number_to_mutate = int(mutation_fraction * len(predicted_values))
        
        # List of metric results from each repetition of the mutation, to be averaged at the end
        mutation_fraction_metrics = []
        
        for _ in range(repetitions):
            # Creating the mutated predictions and retrieving metrics
            indexes_to_mutate = np.random.choice(len(predicted_values), number_to_mutate, replace=False)
            mutated_predictions = np.copy(predicted_values)
            mutated_predictions[indexes_to_mutate] = mutate_to_value
    
            mutation_fraction_metrics.append(get_all_metrics(true_values, mutated_predictions, privilege_status))
        # Average the results across the iterations and append to the results
        mutation_fraction_metrics_averages = dict()
        for metric_type, metric_type_dict in mutation_fraction_metrics[0].items():
            mutation_fraction_metrics_averages[metric_type] = dict()
            
            for metric_name in metric_type_dict.keys():
                mutation_fraction_metrics_averages[metric_type][metric_name] = np.mean([metric[metric_type][metric_name] for metric in mutation_fraction_metrics])
        
        metrics.append((mutation_fraction, mutation_fraction_metrics_averages))
        
    return metrics