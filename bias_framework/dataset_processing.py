import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

def _to_pandas_dataframe(input: ArrayLike):
    """The aif360.datasets require a dataframe argument, but the format of the data after applying the user's preprocessing steps might be a different format. Here we will attempt to convert to a dataframe regardless
    
    Args:
        input (ArrayLike): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    if isinstance(input, pd.DataFrame):
        df_input = input
    elif sparse.issparse(input):
        df_input = pd.DataFrame.sparse.from_spmatrix(input)
    else:
        try:
            df_input = pd.DataFrame(input)
        except:
            raise RuntimeError(f"Pre-processing results in an unrecognised datatype. Please make sure running pre-processing returns a pandas dataframe or a convertible type. If you have used no pre-processing, make sure your data is of the expected type\nEncountered type: {type(input)}")
        
    return df_input
    




def __get_aif360_datasets(self, x_train, x_validation, training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities):
    """We are using debiasing methods implemented in the aif360 library. These require aif360.datasets arguments, so we covert our dataframes to this form. 
    """
    
    # aif360.datasets take a dataframe argument, so make sure this is the type for both x_train and x_validation

    # Finish creating the datasets for training now that we know df_train is a pandas dataframe     
    df_train["Is Privileged"] = self.privilege_train
    
    df_train_true_labels = df_train.copy()
    df_train_true_labels["Class Label"] = self.y_train
    
    train_true_labels = StandardDataset(
        df_train_true_labels, 
        label_name="Class Label", 
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"], 
        privileged_classes=[[1]]
    )
    
    df_train_predictions = df_train.copy()
    df_train_predictions["Probabilities"] = training_probabilities
    df_train_predictions["Class Label"] = training_predicted_values
    train_predictions = StandardDataset(
        df_train_predictions, 
        label_name="Class Label",
        scores_name="Probabilities", 
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"], 
        privileged_classes=[[1]]
    )
    
    # Similarly compute the datasets for the validation set. Note that we don't really need the true labels in any of these datasets
    df_validation["Is Privileged"] = self.privilege_validation 
    
    df_validation_predictions = df_validation.copy()
    df_validation_predictions["Probabilities"] = validation_probabilities
    df_validation_predictions["Class Label"] = validation_predicted_values
    validation_predictions = StandardDataset(
        df_validation_predictions, 
        label_name="Class Label",
        scores_name="Probabilities", 
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"], 
        privileged_classes=[[1]]
    ) 
    
    
    # Some debiasing methodologies need a dataset not only for training but also application. This dataset exists for that purpose, and hides class labels so that the information cannot leak 
    df_validation_to_predict = df_validation.copy()
    df_validation_to_predict["Class Label"] = np.zeros(len(df_validation_to_predict))
    
    validation_to_predict = StandardDataset(
        df_validation_to_predict, 
        label_name="Class Label",
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"], 
        privileged_classes=[[1]]
    ) 
    
    return train_true_labels, train_predictions, validation_predictions, validation_to_predict     