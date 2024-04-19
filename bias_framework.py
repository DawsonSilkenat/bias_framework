import pandas as pd
import numpy as np
from ml_dummy import ML_Model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric



class Bias_Framework:
    def __init__(self, model: ML_Model, df_training_data: pd.DataFrame, df_validation_data: pd.DataFrame) -> None:
        """Creates an instance of the bias framework applied to the specified model and data

        Args:
            model: The ML model to which the bias framework will be applied. This model must have a fit method and predict method. 
            training_data: The data for training the ML model. It is assumed that the last column is the target variable.
            validation_data: The data for which fairness metrics. It is assumed that the columns are the same as training_data.
        """
        if not df_training_data.columns.equals(df_validation_data.columns):
            raise ValueError("The training and validation dataframes must contain the same columns")
        
        self.model = model
        
        target_variable = df_training_data.columns[-1] 

        self.df_y_train = df_training_data[target_variable]
        self.df_x_train = df_training_data.drop(columns=[target_variable])
        
        self.df_y_validate = df_validation_data[target_variable]
        self.df_x_validate = df_validation_data.drop(columns=[target_variable])
        
              
        self.privilege_function = lambda x: False
    
    
    def set_privilege_function(self, function: callable) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class

        Args:
            function: A function which when applied to a row of a dataframe will return a 1 for privileged and 0 for unprivileged. True and false values should also work, but I am less certain of this. Requires testing.
        """
        self.privilege_function = function
 
 
    def set_privileged_combinations(self, privileged_combinations: list[dict[str: any]]) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class by specifying which groupings are considered privileged

        Args:
            privileged_combinations: a list of dictionaries, each dictionary maps a set of columns name to the values they must take. Note: Each columns name must map to a single value, not a list of possible values. The values of any column names not included in the dictionary are ignored
            
            Example from aif360 documentation:
            [{'sex': 1, 'age': 1}, {'sex': 0}]
        """
        self.privilege_function = lambda x: int(
            # Checks if a row matches any of the valid groupings
            any([
                # Checks if all the requirements for a particular grouping are met
                all([x[name] == value for name, value in grouping]) for grouping in privileged_combinations])
        )
    
    
    def set_unprivileged_combinations(self, privileged_combinations: list[dict[str: any]]) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class by specifying which groupings are considered privileged

        Args:
            privileged_combinations: a list of dictionaries, each dictionary maps a set of columns name to the values they must take. Note: Each columns name must map to a single value, not a list of possible values. The values of any column names not included in the dictionary are ignored
            
            Example from aif360 documentation:
            [{'sex': 1, 'age': 1}, {'sex': 0}]
        """
        self.privilege_function = lambda x: int(
            # Checks if a row matches any of the valid groupings, and negates it so we are unprivileged
            not any([
                # Checks if all the requirements for a particular grouping are met
                all([x[name] == value for name, value in grouping]) for grouping in privileged_combinations])
        )
        
    
    def get_error_metrics(self, y_true_value: np.array, y_predicted_value: np.array) -> dict[str, float]:
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
        metrics["recall positive class"]            = recall_score(y_true_value, y_predicted_value, pos_label=1)
        metrics["recall negative class"]            = recall_score(y_true_value, y_predicted_value, pos_label=0)
        metrics["recall macro average"]             = recall_score(y_true_value, y_predicted_value, average="macro")
        metrics["precision positive class"]         = precision_score(y_true_value, y_predicted_value, pos_label=1)
        metrics["precision negative class"]         = precision_score(y_true_value, y_predicted_value, pos_label=0)
        metrics["precision macro average"]          = precision_score(y_true_value, y_predicted_value, average="macro")
        metrics["f1 score positive class"]          = f1_score(y_true_value, y_predicted_value, pos_label=1)
        metrics["f1 score negative class"]          = f1_score(y_true_value, y_predicted_value, pos_label=0)
        metrics["f1 score macro average"]           = f1_score(y_true_value, y_predicted_value, average="macro")
        metrics["Matthews correlation coefficient"] = matthews_corrcoef(y_true_value, y_predicted_value)
    
        return metrics
    
    
    def get_fairness_metrics(df: pd.DataFrame, true_values: np.array, predicted_values: np.array, privilege_function: callable) -> dict[str, float]:
        
        df["Is Privileged"] = df.apply(privilege_function, axis=1)
        
        df_dataset_with_true_class = df.copy()
        df_dataset_with_true_class["Class Label"] = true_values
        
        df_dataset_with_predicted_class = df.copy()
        df_dataset_with_predicted_class["Class Label"] = predicted_values
        
        # I don't believe the values of protected_attribute_names and privileged_classes matter here since this is really set in ClassificationMetric, however these are required fields so might as well set them reasonably. 
        dataset_with_true_class = StandardDataset(df_dataset_with_true_class, 
                          label_name="Class Label", 
                          favorable_classes=[1],
                          protected_attribute_names=["Is Privileged"], 
                          privileged_classes=[1]
                          )
        
        dataset_with_predicted_class = StandardDataset(df_dataset_with_true_class, 
                          label_name="Class Label", 
                          favorable_classes=[1],
                          protected_attribute_names=["Is Privileged"], 
                          privileged_classes=[1]
                          )
        
    
        class_metric = ClassificationMetric(dataset_with_true_class,dataset_with_predicted_class, unprivileged_groups=[{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}])
    
        metrics = dict()
        metrics["statistical parity difference"] = abs(class_metric.statistical_parity_difference())
        metrics["average odds difference"]       = abs(class_metric.average_abs_odds_difference())
        metrics["equal opportunity difference"]  = abs(class_metric.equal_opportunity_difference())
        metrics["error_rate_difference"]         = abs(class_metric.error_rate_difference())
        
        return metrics


    def fairea_model_mutation(self, df: pd.DataFrame, true_values: np.array, predicted_values: np.array, privilege_function: callable, mutation_percentages: list[float] = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]):
        pass