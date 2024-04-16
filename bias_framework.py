import pandas as pd
import numpy as np

class ML_Model:
    """Dummy class representing our expectations regarding LM models for which this framework should function. 
    We assume binary classifier at the moment, I believe longterm it would be good to expand this. 
    """
    def fit(training_data: pd.DataFrame, target_values: pd.DataFrame) -> None:
        raise NotImplementedError
    
    def predict(to_classify: pd.DataFrame) -> np.array:
        raise NotImplementedError


class Bias_Framework:
    def __init__(self, model: ML_Model, df_training_data: pd.DataFrame, df_validation_data: pd.DataFrame, **kwargs: dict) -> None:
        """Creates an instance of the bias framework applied to the specified model and data

        Args:
            model: The ML model to which the bias framework will be applied. This model must have a fit method and predict method. 
            training_data: The data for training the ML model. It is assumed that the last column is the target variable.
            validation_data: The data for which fairness metrics. It is assumed that the columns are the same as training_data.
            **kwargs: Any model hyperparameters
        """
        self.model = model
        
        target_variable = df_training_data.columns[-1] 

        self.df_y_train = df_training_data[target_variable]
        self.df_x_train = df_training_data.drop(columns=[target_variable])
        
        self.df_y_validate = df_validation_data[target_variable]
        self.df_x_validate = df_validation_data.drop(columns=[target_variable])
        
        self.hyperparameters = kwargs
    
    
    
    
    def get_model(self, version):
        pass
    
    
    
    