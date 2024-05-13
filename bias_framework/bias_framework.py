import pandas as pd
import numpy as np
from .metrics import *
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import LFR, Reweighing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from sklearn.calibration import CalibratedClassifierCV
from scipy import sparse
from .debiasing_graphs import DebiasingGraphsObject
from .baselines.fairea_curve import FaireaCurve

# This is only for information on runtime rather than used functionally
import time

# aif360 seems to do something pandas doesn't like, and it makes it hard to debug when all I read are these warning messages
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class Bias_Framework:
    def __init__(self, model, df_x_train: pd.DataFrame, df_x_validation: pd.DataFrame, df_y_train: pd.DataFrame, df_y_validation: pd.DataFrame,pre_processing=None, **model_args) -> None:
        """Creates an instance of the bias framework applied to the specified model and data

        Args:
            model: The ML model to which the bias framework will be applied. This model must have a fit method and predict method. I am assuming at the moment that this will be an sklearn ml model, might try to modify this to be more flexible later.
            df_x_train (pd.DataFrame): The training data features for the ML model
            df_x_validation (pd.DataFrame): The validation data features for the ML model
            df_y_train (pd.DataFrame): The training data labels for the ML model
            df_y_validation (pd.DataFrame): The validation data labels for the ML model
            pre_processing (optional): Any pre-processing to apply to the data before use by the ml model. Expected to implement methods for fit_transform and transform.
        """

        
        # Not all models implement a function for prediction probability. Probabilities are required for some postprocessing debiasing. The CalibratedClassifierCV method preforms probability calibration, which also adds this functionality if it is not part of the model
        # TODO create class which defines the required interface, remove this line of code and place responsibility on user to satisfy interface
        self.model = CalibratedClassifierCV(estimator=model)

        self.df_x_train = df_x_train
        self.df_x_validation = df_x_validation
        
        self.y_train = df_y_train.to_numpy().ravel()
        self.y_validation = df_y_validation.to_numpy().ravel()
        
        self.privilege_train = np.ones(len(df_x_train))
        self.privilege_validation = np.ones(len(df_y_train))
        
        self.pre_processing = pre_processing
        
        self.__model_args = model_args
        self.__fairea = None
        self.__metrics_by_debiasing_technique = dict()
    
    
    def set_privilege_function(self, privilege_function: callable) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class

        Args:
            function: A function which when applied to a row of a dataframe will return a 1 for privileged and 0 for unprivileged. True and false values should also work.
        """
        
        self.privilege_train = self.df_x_train.apply(privilege_function, axis=1).to_numpy()
        self.privilege_validation = self.df_x_validation.apply(privilege_function, axis=1).to_numpy()
 
 
    def set_privileged_combinations(self, privileged_combinations: list[dict[str: any]]) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class by specifying which groupings are considered privileged

        Args:
            privileged_combinations: a list of dictionaries, each dictionary maps a set of columns name to the values they must take. Note: Each columns name must map to a single value, not a list of possible values. The values of any column names not included in the dictionary are ignored
            
            Example from aif360 documentation:
            [{'sex': 1, 'age': 1}, {'sex': 0}]
            The first dictionary indicates that if sex has value 1 and age has value 1 then the individual belongs to the privileged group
            The second dictionary indicates that if sex has value 0 then the individual belongs to the privileged group (regardless of age)
        """
        privilege_function = lambda x: int(
            # Checks if a row matches any of the valid groupings
            any([
                # Checks if all the requirements for a particular grouping are met
                all([x[name] == value for name, value in grouping]) for grouping in privileged_combinations])
        )
        self.set_privilege_function(privilege_function)
    
    
    def set_unprivileged_combinations(self, unprivileged_combinations: list[dict[str: any]]) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class by specifying which groupings are not considered privileged

        Args:
            unprivileged_combinations: a list of dictionaries, each dictionary maps a set of columns name to the values they must take. Note: Each columns name must map to a single value, not a list of possible values. The values of any column names not included in the dictionary are ignored
            
            Example from aif360 documentation:
            [{'sex': 1, 'age': 1}, {'sex': 0}]
            The first dictionary indicates that if sex has value 1 and age has value 1 then the individual belongs to the unprivileged group
            The second dictionary indicates that if sex has value 0 then the individual belongs to the unprivileged group (regardless of age)
        """
        privilege_function = lambda x: int(
            # Checks if a row matches any of the valid groupings, and negates it so we are unprivileged
            not any([
                # Checks if all the requirements for a particular grouping are met
                all([x[name] == value for name, value in grouping]) for grouping in unprivileged_combinations])
        )  
        self.set_privilege_function(privilege_function)
    
    
    def run_framework(self):     
        """Executes the framework using the model, data, and pre-processing provided at initialisation and the definition of privilege set using one of the provided methods. This will run through a number of debiasing methodologies and save the results. This may take some time. Once finished, you may either call one of the graph displaying methods of this class or the get_FaireaGraphsObject method to get an object which stores just the result and can be added to similar objects to combine the graphs.
        """   
        
        # When this code is run we must assume the user has already assigned privilege. We can therefore apply the pre-processing step without losing information
        x_train = self.df_x_train
        if self.pre_processing:
            x_train = self.pre_processing.fit_transform(x_train, self.y_train)
        x_validation = self.df_x_validation
        if self.pre_processing:
            x_validation = self.pre_processing.transform(x_validation)
        
        start = time.time()
        training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities = self.__no_debiasing(x_train, x_validation)
        print(f"{time.time() - start} seconds to run with no debiasing")
        
        start = time.time()
        fairea_curve = FaireaCurve(self.y_validation, validation_predicted_values, self.privilege_validation)
        print(f"{time.time() - start} seconds to get fairea baseline")
         
        train_true_labels, train_predictions, validation_predictions, validation_to_predict = self.__get_aif360_datasets(x_train, x_validation, training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities)
        
        start = time.time()
        # self.__learning_fair_representation(train_true_labels, validation_to_predict) 
        print(f"{time.time() - start} seconds to run learning fair representation")
        
        start = time.time()
        self.__reweighing(train_true_labels, validation_to_predict)
        print(f"{time.time() - start} seconds to run reweighing")
        
        start = time.time()
        # self.__reject_option_classification(train_true_labels, train_predictions, validation_predictions)
        print(f"{time.time() - start} seconds to run reject option classification")
        
        start = time.time()
        self.__calibrated_equal_odds(train_true_labels, train_predictions, validation_predictions)
        print(f"{time.time() - start} seconds to run calibrated equal odds")
        
        start = time.time()
        self.__equal_odds(train_true_labels, train_predictions, validation_predictions)
        print(f"{time.time() - start} seconds to run equal odds")
        
        self.__fairea = DebiasingGraphsObject(self.__metrics_by_debiasing_technique, fairea_curve)
      
        
    def get_debias_methodologies(self) -> list[str]:
        if self.__fairea is None:
            return []
        return self.__fairea.get_debias_methodologies()
    
    
    def get_error_metric_names(self) -> list[str]:
        if self.__fairea is None:
            return []
        return self.__fairea.get_error_metric_names()      
     
                 
    def get_bias_metric_names(self) -> list[str]:
        if self.__fairea is None:
            return []
        return self.__fairea.get_bias_metric_names()    
    
    
    def get_raw_data(self) -> dict[str, dict[str, dict[str, float]]]:
        if self.__fairea is None:
            return dict()
        return self.__fairea.get_raw_data()    
    
    
    def get_FaireaGraphsObject(self) -> DebiasingGraphsObject:
        return self.__fairea  
    
    
    def show_fairea_graph(self, error_metric: str, fairness_metric: str) -> None:
        self.__fairea.show_single_graph(error_metric, fairness_metric)
    
    
    def show_many_fairea_graphs(self, error_metrics: list[str], fairness_metrics: list[str]) -> None:
        self.__fairea.show_subplots(error_metrics, fairness_metrics)
        
    
    def show_all_fairea_graphs(self) -> None:
        self.__fairea.show_all_subplots()
        
        
    def __get_aif360_datasets(self, x_train, x_validation, training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities):
        # To avoid recomputing for each debiasing methodology we create the aif360.datasets and pass them to each function
        # aif360.datasets take a dataframe argument, so make sure this is the type for both x_train and x_validation
        if isinstance(x_train, pd.DataFrame):
            df_train = x_train
            df_validation = x_validation
        elif sparse.issparse(x_train):
            df_train = pd.DataFrame.sparse.from_spmatrix(x_train)
            df_validation = pd.DataFrame.sparse.from_spmatrix(x_validation)
        elif isinstance(x_train, np.ndarray):
            df_train = pd.DataFrame(x_train)
            df_validation = pd.DataFrame(x_validation)
        else:
            raise RuntimeError(f"Pre-processing results in an unrecognised datatype. Please make sure running pre-processing returns a pandas dataframe or a numpy array. If you have used no pre-processing, make sure your data is of the expected type\nEncountered type: {type(x_train)}")
        
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
    
    
    def __no_debiasing(self, x_train, x_validation):
        self.model.fit(x_train, self.y_train, **self.__model_args)
        
        training_predicted_values = self.model.predict(x_train)
        training_probabilities = self.model.predict_proba(x_train)[:, 1]
                   
        validation_predicted_values = self.model.predict(x_validation)
        validation_probabilities = self.model.predict_proba(x_validation)[:, 1]
        
        self.__metrics_by_debiasing_technique["no debiasing"] = bootstrap_all_metrics(self.y_validation, validation_predicted_values, self.privilege_validation)
        self.__metrics_by_debiasing_technique["no debiasing"]["raw"] = validation_predicted_values
        
        # Most debiasing analysis functions won't return anything, however having default results is needed for fairea and postprocessing debiasing 
        return training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities
   
        
    # def __optimized_preprocessing(self):
        # this looks like it requires more knowledge of the dataset than I expect this framework to know at this point in time, so leaving this unimplemented
        # Refer back to these when it comes time to implement
        # from aif360.algorithms.preprocessing import OptimPreproc
        # from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
        pass
    
    
    def __learning_fair_representation(self, train_true_labels, validation_to_predict):
        # Only running one number_of_prototypes due to runtime. When ran with [5, 10] this took over an hour while other methods took minutes.
        for number_of_prototypes in [5]:
            training_dataset = train_true_labels.copy()

            # Applying learning fair representation to the training data
            fair_representation = LFR(unprivileged_groups=[{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}], k=number_of_prototypes)
            
            transformed_data = fair_representation.fit_transform(training_dataset)
            
            # This debiasing methodology also mutates the labels. If there is a single label,the classifier may raise an exception, so this case must be handled separately
            classes = np.unique(transformed_data.labels.ravel())
            if len(classes) == 1:
                # It is reasonable to assume that if only one class exists in the training data, it will be the only predicted value.
                predicted_values = np.full(len(self.y_validation), classes[0])
            else:
                # Note that this debiasing methodology also seems to update the class labels. I'm not entirely clear on why, but does seem to get better results with the updated labels.
                self.model.fit(transformed_data.features, transformed_data.labels.ravel(), **self.__model_args)

                validation_dataset = validation_to_predict.copy()
                validation_dataset = fair_representation.transform(validation_dataset)
                predicted_values = self.model.predict(validation_dataset.features)
            

            self.__metrics_by_debiasing_technique[f"learning fair representation with {number_of_prototypes} prototypes"] = bootstrap_all_metrics(self.y_validation, predicted_values, self.privilege_validation)
            self.__metrics_by_debiasing_technique[f"learning fair representation with {number_of_prototypes} prototypes"]["raw"] = predicted_values


    def __reweighing(self, train_true_labels, validation_to_predict):
        training_dataset = train_true_labels.copy()
        
        # Applying reweighing to the training data
        reweighing = Reweighing(unprivileged_groups=[{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}])
        transformed_data = reweighing.fit_transform(training_dataset)
        self.model.fit(transformed_data.features, transformed_data.labels.ravel(), sample_weight=transformed_data.instance_weights, **self.__model_args)
        
        # Applying the required modifications to the validation data and getting results for metric calculation
        # Note that we don't need to apply reweighing because that only impacts the training stage
        predicted_values = self.model.predict(validation_to_predict.copy().features)
        
        self.__metrics_by_debiasing_technique["reweighing"] = bootstrap_all_metrics(self.y_validation, predicted_values, self.privilege_validation)
        self.__metrics_by_debiasing_technique["reweighing"]["raw"] = predicted_values
    
    
    def __reject_option_classification(self, train_true_labels, train_predictions, validation_predicted_values):
        # We want results for optimising with respect to each of the bias metrics, as it is unclear what the trade off will be
        for metric_name in ["Statistical parity difference", "Average odds difference", "Equal opportunity difference"]:
            training_dataset_true = train_true_labels.copy()
            training_dataset_predictions = train_predictions.copy()

            reject_option_classification = RejectOptionClassification([{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}], metric_name=metric_name)
            reject_option_classification.fit(training_dataset_true, training_dataset_predictions)
            
            validation_dataset = validation_predicted_values.copy()
            predicted_values = reject_option_classification.predict(validation_dataset).labels.ravel()

            self.__metrics_by_debiasing_technique[f"reject option classification {metric_name.lower()} optimised"] = bootstrap_all_metrics(self.y_validation, predicted_values, self.privilege_validation)     
            self.__metrics_by_debiasing_technique[f"reject option classification {metric_name.lower()} optimised"]["raw"] = predicted_values   


    def __calibrated_equal_odds(self, train_true_labels, train_predictions, validation_predicted_values):
        for cost_constraint in ["fpr", "fnr", "weighted"]:

            training_dataset_true = train_true_labels.copy()
            training_dataset_predictions = train_predictions.copy()
            
            calibrated_equal_odds = CalibratedEqOddsPostprocessing([{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}], cost_constraint=cost_constraint)
            calibrated_equal_odds.fit(training_dataset_true, training_dataset_predictions)
            
            validation_dataset = validation_predicted_values.copy()
            predicted_values = calibrated_equal_odds.predict(validation_dataset).labels.ravel()
            
            self.__metrics_by_debiasing_technique[f"calibrated equal odds using {cost_constraint} cost"] = bootstrap_all_metrics(self.y_validation, predicted_values, self.privilege_validation)  
            self.__metrics_by_debiasing_technique[f"calibrated equal odds using {cost_constraint} cost"]["raw"] = predicted_values         


    def __equal_odds(self, train_true_labels, train_predictions, validation_predicted_values):
        training_dataset_true = train_true_labels.copy()
        training_dataset_predictions = train_predictions.copy()
        
        calibrated_equal_odds = EqOddsPostprocessing([{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}])
        calibrated_equal_odds.fit(training_dataset_true, training_dataset_predictions)
        
        validation_dataset = validation_predicted_values.copy()
        predicted_values = calibrated_equal_odds.predict(validation_dataset).labels.ravel()
        
        self.__metrics_by_debiasing_technique["equal odds"] = bootstrap_all_metrics(self.y_validation, predicted_values, self.privilege_validation)
        self.__metrics_by_debiasing_technique["equal odds"]["raw"] = predicted_values 
        
