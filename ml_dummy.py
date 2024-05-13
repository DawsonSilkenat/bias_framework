import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin 

class ML_Model(ClassifierMixin):
    """Dummy class representing our expectations regarding LM models for which this framework should function. 
    We assume binary classifier at the moment, I believe longterm it would be good to expand this. 
    """
    def fit(training_data: pd.DataFrame, target_values: pd.DataFrame, sample_weight: np.array) -> None:
        raise NotImplementedError
    
    def predict(to_classify: pd.DataFrame) -> np.array:
        raise NotImplementedError