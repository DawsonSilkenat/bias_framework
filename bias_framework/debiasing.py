from .metrics import bootstrap_all_metrics

def no_debiasing(model, x_train, x_validation, y_train, y_validation, privilege_validation):
    """We require the base models predictions in order to apply debiasing techniques 

    Args:
        model: The ML model to which we wish to apply debiasing
        x_train: _description_
        x_validation (_type_): _description_
        y_train (_type_): _description_

    Returns:
        tuple: tuple of two elements
            first element (np.array, np.array, np.array, np.array): The model predictions and probability of positive class for training dataset and validation dataset respectively
            second element (dict[str, dict[str, dict[str, float]]]): dictionary of metric type -> metric name -> summary statistic
    """
    
    model.fit(x_train, y_train)
    
    # Predicted class label and probability of positive class for both the training and validation sets are required for debiasing methods
    training_predicted_values = model.predict(x_train)
    training_probabilities = model.predict_proba(x_train)[:, 1]
    validation_predicted_values = model.predict(x_validation)
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    
    base_model_predictions = (training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities)
        
    metric_results = bootstrap_all_metrics(y_validation, validation_predicted_values, privilege_validation)   
    # metric_results["raw"] = validation_predicted_values
    
    return base_model_predictions, metric_results