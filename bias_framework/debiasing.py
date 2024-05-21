import numpy as np
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import LFR, Reweighing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing


def no_debiasing(model, x_train, x_validation, y_train) -> tuple[np.array, np.array, np.array, np.array]:
    """We require the base models predictions in order to apply debiasing techniques 

    Args:
        model: The ML model to which we wish to apply debiasing. We assume that this model is capable of both providing predictions and assigning probabilities to those predictions
        x_train: Training data, as formatted by any pre-processing required by the model
        x_validation: Validation data, as formatted by any pre-processing required by the model
        y_train: Training target variable
        
    Returns:
        (np.array, np.array, np.array, np.array): The model predictions and probability of positive class for training dataset and validation dataset respectively
    """
    
    model.fit(x_train, y_train)
    
    # Predicted class label and probability of positive class for both the training and validation sets are required for debiasing methods
    training_predicted_values = model.predict(x_train)
    training_probabilities = model.predict_proba(x_train)[:, 1]
    validation_predicted_values = model.predict(x_validation)
    validation_probabilities = model.predict_proba(x_validation)[:, 1]    
    
    return training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities 

# TODO all of the summaries could include a more complete description of how the debiasing methodology works

def learning_fair_representation(model, ds_train_true_labels: StandardDataset, ds_validation_to_predict: StandardDataset, number_of_prototypes: list[int] = None, seed: int = None) -> dict[str, np.array]:
    """Apply the learning fair representation debiasing methodology and return the predictions

    Args:
        model: The ML model to which we wish to apply debiasing
        ds_train_true_labels (StandardDataset): The dataset containing the training data and target labels
        ds_validation_to_predict (StandardDataset): The dataset containing the validation data
        number_of_prototypes (list[int], optional): Each element is an application of learning fair representation using the specified number of prototypes. Defaults to [5].
        seed (int, optional): Random seed for consistent results. Defaults to None.

    Returns:
        dict[str, np.array]: Maps a string, including how many prototypes were used, to the model predictions
    """
    if number_of_prototypes is None:
        number_of_prototypes = [5]
    elif isinstance(number_of_prototypes, int):
        number_of_prototypes = [number_of_prototypes]
    
    results = dict()
    
    for k in number_of_prototypes:
        # Applying learning fair representation to the training data
        fair_representation = LFR(unprivileged_groups=[{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}], k=k, seed=seed)
        ds_train_transformed = fair_representation.fit_transform(ds_train_true_labels)
         
        # This debiasing methodology also mutates the labels. If there is a single label, the classifier may raise an exception. This case must be handled separately
        classes = np.unique(ds_train_transformed.labels.ravel())
        if len(classes) == 1:
            # It is reasonable to assume that if only one class exists in the training data, it will be the only predicted value.
            predicted_values = np.full(len(ds_train_transformed.labels.ravel()), classes[0])
        else:
            # I'm not entirely clear on why, but this methodology seem to get better results with the updated labels. I should read more and attempt to leave a comment explaining why. If you are reading this, here is the link to the paper: http://proceedings.mlr.press/v28/zemel13.pdf
            model.fit(ds_train_transformed.features, ds_train_transformed.labels.ravel())
            ds_validation_transformed = fair_representation.transform(ds_validation_to_predict)
            predicted_values = model.predict(ds_validation_transformed.features)
        
        results[f"learning fair representation with {k} prototypes"] = predicted_values

    return results


def reweighting(model, ds_train_true_labels: StandardDataset, ds_validation_to_predict: StandardDataset) -> dict[str, np.array]:
    """Apply the reweighting debiasing methodology and return the predictions

    Args:
        model: The ML model to which we wish to apply debiasing
        ds_train_true_labels (StandardDataset): The dataset containing the training data and target labels
        ds_validation_to_predict (StandardDataset): The dataset containing the validation data

    Returns:
        dict[str, np.array]: Maps the string 'reweighting' to the model predictions
    """
    results = dict()
    reweighing = Reweighing(unprivileged_groups=[{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}])
    ds_transformed_data = reweighing.fit_transform(ds_train_true_labels)
    
    model.fit(ds_transformed_data.features, ds_transformed_data.labels.ravel(), sample_weight=ds_transformed_data.instance_weights)    
    results["reweighting"] = model.predict(ds_validation_to_predict.features)
    
    return results


def reject_option_classification(ds_train_true_labels: StandardDataset, ds_train_predictions: StandardDataset, ds_validation_predictions: StandardDataset, bias_metrics: list[str] = None) -> dict[str, np.array]:
    """Apply the reject option classification debiasing methodology and return the predictions

    Args:
        ds_train_true_labels (StandardDataset): The dataset containing the training data and target labels
        ds_train_predictions (StandardDataset): The dataset containing the training data and model predictions. Must include the probabilities of said predictions.
        ds_validation_predictions (StandardDataset): The dataset containing the validation data and model predictions. Must include the probabilities of said predictions.
        bias_metrics (list[str], optional): Reject option classification optimises with respect to a specified measure of bias. This will be preformed for every metric specified in this list. The options are "Statistical parity difference", "Average odds difference", and "Equal opportunity difference". Defaults to ["Statistical parity difference", "Average odds difference", "Equal opportunity difference"].

    Returns:
        dict[str, np.array]: Maps a string, including the metric used, to the model predictions
    """
    if bias_metrics is None:
        bias_metrics = ["Statistical parity difference", "Average odds difference", "Equal opportunity difference"]
    elif isinstance(bias_metrics, str):
        bias_metrics = [bias_metrics]  
    
    # The allowed names must be an exact match. To make this easier we handle capitalisation here 
    bias_metrics = [metric_name.capitalize() for metric_name in bias_metrics]
    
    results = dict()
    
    for metric_name in bias_metrics:
        reject_option_classification = RejectOptionClassification([{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}], metric_name=metric_name)
        reject_option_classification.fit(ds_train_true_labels, ds_train_predictions)
            
        results[f"reject option classification {metric_name.lower()} optimised"] = reject_option_classification.predict(ds_validation_predictions).labels.ravel()

    return results  


def calibrated_equal_odds(ds_train_true_labels: StandardDataset, ds_train_predictions: StandardDataset, ds_validation_predictions: StandardDataset, cost_constraints: list[str] = None, seed: int = None) -> dict[str, np.array]:
    """Apply the calibrated equal odds debiasing methodology and return the predictions

    Args:
        ds_train_true_labels (StandardDataset): The dataset containing the training data and target labels
        ds_train_predictions (StandardDataset): The dataset containing the training data and model predictions. Must include the probabilities of said predictions.
        ds_validation_predictions (StandardDataset): The dataset containing the validation data and model predictions. Must include the probabilities of said predictions.
        cost_constraints (list[str], optional): Calibrated equal odds optimises with respect to a specified measure of cost. This will be preformed for every metric specified in this list. The options are "fpr", "fnr", and "weighted". Defaults to ["fpr", "fnr", "weighted"].
        seed (int, optional): Random seed for consistent results. Defaults to None. 

    Returns:
        dict[str, np.array]: Maps a string, including the cost used ,to the model predictions
    """
    if cost_constraints is None:
        cost_constraints = ["fpr", "fnr", "weighted"]
    elif isinstance(cost_constraints, str):
        cost_constraint = [cost_constraints]
    
    results = dict()
    
    for cost_constraint in cost_constraints:            
        calibrated_equal_odds = CalibratedEqOddsPostprocessing([{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}],cost_constraint=cost_constraint, seed=seed)
        calibrated_equal_odds.fit(ds_train_true_labels, ds_train_predictions)
            
        results[f"calibrated equal odds using {cost_constraint} cost"] = calibrated_equal_odds.predict(ds_validation_predictions).labels.ravel()
            
    return results    


def equal_odds(ds_train_true_labels: StandardDataset, ds_train_predictions: StandardDataset, ds_validation_predictions: StandardDataset, seed: int = None) -> dict[str, np.array]:
    """Apply the equal odds debiasing methodology and return the predictions

    Args:
        ds_train_true_labels (StandardDataset): The dataset containing the training data and target labels
        ds_train_predictions (StandardDataset): The dataset containing the training data and model predictions. Must include the probabilities of said predictions.
        ds_validation_predictions (StandardDataset): The dataset containing the validation data and model predictions. Must include the probabilities of said predictions.
        seed (int, optional): Random seed for consistent results. Defaults to None. 

    Returns:
        dict[str, np.array]: Maps the string 'equal odds' to the model predictions
    """
    results = dict()
    calibrated_equal_odds = EqOddsPostprocessing([{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}], seed=seed)
    calibrated_equal_odds.fit(ds_train_true_labels, ds_train_predictions)
    results["equal odds"] = calibrated_equal_odds.predict(ds_validation_predictions).labels.ravel()
    
    return results