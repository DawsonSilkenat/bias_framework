import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from aif360.datasets import StandardDataset


def _to_pandas_dataframe(input: ArrayLike) -> pd.DataFrame:
    """The aif360.datasets require a dataframe argument, but the format 
    of the data after applying the user's preprocessing steps might be a 
    different format. Here we will attempt to convert to a dataframe 
    regardless
    """
    if isinstance(input, pd.DataFrame):
        df_input = input
    elif sparse.issparse(input):
        df_input = pd.DataFrame.sparse.from_spmatrix(input)
    else:
        try:
            df_input = pd.DataFrame(input)
        except:
            raise RuntimeError(
                ("Pre-processing results in an unrecognised datatype. Please"
                 "make sure running pre-processing returns a pandas dataframe" 
                 "or a convertible type. If you have used no pre-processing," 
                 "make sure your data is of the expected type\nEncountered" 
                 f"type: {type(input)}"))

    return df_input


def covert_to_datasets_train(
        x_train: ArrayLike, y_train: np.array, train_predictions: np.array, 
        train_probabilities: np.array, privilege_train: np.array
        ) -> tuple[StandardDataset, StandardDataset]:
    """Converts the training data into aif360.datasets StandardDataset 
    objects. Objects of this type are required for the debiasing methods 
    implemented in aif360. These datasets are primarily used in 
    pre-processing debiasing methodologies. I personally find these a 
    bit unintuitive to work with, so have attempted to not require the 
    user consider their implementation.

    Args:
        x_train (ArrayLike): The training data, after any preprocessing 
        is applied. 
        y_train (np.array): The training labels
        train_predictions (np.array): The class labels assigned to 
        x_train by the ml model we wish to debias
        train_probabilities (np.array): The probabilities of the correct 
        class label being the positive class, as assigned by the ml 
        model we wish to debias. 
        privilege_train (np.array): Binary array, the ith entry of which 
        is a 1 if the ith element of the training data belongs to the 
        privileged group, and 0 otherwise

    Returns:
        tuple[StandardDataset, StandardDataset]: 
            The first dataset (ds_train_true_labels) contains the 
            training data, privilege statuses, and true classes
            The second dataset (ds_train_predictions) contains the 
            training data, privilege statuses, probability of belonging 
            to the privileged class, and predicted class label
    """
    df_x_train = _to_pandas_dataframe(x_train)
    df_x_train["Is Privileged"] = privilege_train

    df_train_true_labels = df_x_train.copy()
    df_train_true_labels["Class Label"] = y_train

    ds_train_true_labels = StandardDataset(
        df_train_true_labels,
        label_name="Class Label",
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"],
        privileged_classes=[[1]]
    )

    df_train_predictions = df_x_train.copy()
    df_train_predictions["Probabilities"] = train_probabilities
    df_train_predictions["Class Label"] = train_predictions

    ds_train_predictions = StandardDataset(
        df_train_predictions,
        label_name="Class Label",
        scores_name="Probabilities",
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"],
        privileged_classes=[[1]]
    )

    return ds_train_true_labels, ds_train_predictions


def covert_to_datasets_validation(
        x_validation: ArrayLike, validation_predictions: np.array, 
        validation_probabilities: np.array, privilege_validation: np.array
        ) -> tuple[StandardDataset, StandardDataset]:
    """Converts the validation data into aif360.datasets StandardDataset 
    objects. Objects of this type are required for the debiasing methods 
    implemented in aif360. These datasets are primarily used in 
    post-processing debiasing methodologies. I personally find these a 
    bit unintuitive to work with, so have attempted to not require the 
    user consider their implementation.

    Args:
        x_validation (ArrayLike): The validation data, after any 
        preprocessing is applied. 
        validation_predictions (np.array): The class labels assigned to 
        x_validation by the ml model we wish to debias
        validation_probabilities (np.array): The probabilities of the 
        correct class label being the positive class, as assigned by the 
        ml model we wish to debias. 
        privilege_validation (np.array): Binary array, the ith entry of 
        which is a 1 if the ith element of the validation data belongs 
        to the privileged group, and 0 otherwise

    Returns:
        tuple[StandardDataset, StandardDataset]: 
            The first dataset (ds_validation_predictions) contains the 
            validation data, privilege statuses, probability of 
            belonging to the privileged class, and predicted class 
            label.
            The second dataset (ds_validation_to_predict) contains the 
            validation data and privilege statuses. A class label is 
            also required, which we set entirely to zeros to prevent 
            information leakage. 
    """
    df_x_validation = _to_pandas_dataframe(x_validation)
    df_x_validation["Is Privileged"] = privilege_validation

    df_validation_predictions = df_x_validation.copy()
    df_validation_predictions["Probabilities"] = validation_probabilities
    df_validation_predictions["Class Label"] = validation_predictions

    ds_validation_predictions = StandardDataset(
        df_validation_predictions,
        label_name="Class Label",
        scores_name="Probabilities",
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"],
        privileged_classes=[[1]]
    )

    # Some debiasing methodologies need a dataset not only for training 
    # but also application. This dataset exists for that purpose, and 
    # hides class labels so that the information cannot leak
    df_validation_no_labels = df_x_validation.copy()
    df_validation_no_labels["Class Label"] = np.zeros(
        len(df_validation_no_labels))

    ds_validation_to_predict = StandardDataset(
        df_validation_no_labels,
        label_name="Class Label",
        favorable_classes=[1],
        protected_attribute_names=["Is Privileged"],
        privileged_classes=[[1]]
    )

    return ds_validation_predictions, ds_validation_to_predict


def get_features_without_privileged_status(
    ds_get_features_from: StandardDataset) -> np.array:
    return np.delete(ds_get_features_from.features, 
                     ds_get_features_from.feature_names.index("Is Privileged"), 
                     axis=1)
