import pandas as pd
import numpy as np
from .metrics import *
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import LFR, Reweighing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from sklearn.calibration import CalibratedClassifierCV
from scipy import sparse

# I quite like the idea of Discrimination aware Ensemble, but it doesn't work with arbitrary classifiers. Might be something to look into later

# This is only for information on runtime rather than used functionally
import time

# aif360 seems to do something pandas doesn't like, and it makes it hard to debug when all I read are these warning messages
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Will have problems if the data is pasted before pre-processing, ie strings must be encoded as numbers before passed to the framework
# Actually, this causes problems as well since pre-processing will likely be arrays and not dataframes
# Can't modify the dataframe since the ml model might have preprocessing which depends on the dataframe structure or drops features

# Could take the pre-processing, find the bias from the original dataframe, transform the data, add the bias information, then applying the debiasing methodology
class Bias_Framework:
    def __init__(self, model, df_x_train: pd.DataFrame, df_x_validation: pd.DataFrame, df_y_train: pd.DataFrame, df_y_validation: pd.DataFrame,pre_processing=None) -> None:
        """Creates an instance of the bias framework applied to the specified model and data

        Args:
            model: The ML model to which the bias framework will be applied. This model must have a fit method and predict method. I am assuming at the moment that this will be an sklearn ml model, might try to modify this to be more flexable later.
            training_data: The data for training the ML model. It is assumed that the last column is the target variable.
            validation_data: The data for which fairness metrics. It is assumed that the columns are the same as training_data.
        """
        
        # Not all models implement a function for prediction probability. Probabilities are required for some postprocessing debiasing. The CalibratedClassifierCV method preforms probability calibration, which also adds this functionality if it is not part of the model
        self.model = CalibratedClassifierCV(estimator=model)

        self.df_x_train = df_x_train
        self.df_x_validation = df_x_validation
        self.df_y_train = df_y_train
        self.df_y_validation = df_y_validation
        
        self.privilege_train = np.ones(len(df_x_train))
        self.privilege_validation = np.ones(len(df_y_train))
        
        self.pre_processing = pre_processing
        
        self.__fairea = None
        self.__metrics_by_debiasing_technique = dict()
    
    
    def set_privilege_function(self, privilege_function: callable) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class

        Args:
            function: A function which when applied to a row of a dataframe will return a 1 for privileged and 0 for unprivileged. True and false values should also work, but I am less certain of this. Requires testing.
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
        # When this code is run we must assume the user has already asigned privilege. We can therefore apply the pre-processing step without losing information
        x_train = self.df_x_train
        if self.pre_processing:
            x_train = self.pre_processing.fit_transform(x_train, self.df_y_train)
        x_validation = self.df_x_validation
        if self.pre_processing:
            x_validation = self.pre_processing.transform(x_validation)
        
        start = time.time()
        training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities = self.__no_debiasing(x_train, x_validation)
        print(f"{time.time() - start} seconds to run with no debiasing")
        
        start = time.time()
        self.__fairea = fairea_model_mutation(self.df_y_validation.to_numpy(), validation_predicted_values, self.privilege_validation)
        print(f"{time.time() - start} seconds to get fairea baseline")
         
        train_true_labels, train_predictions, validation_predictions, validation_to_predict = self.__get_aif360_datasets(x_train, x_validation, training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities)
        
        start = time.time()
        self.__learning_fair_representation(train_true_labels, validation_to_predict) 
        print(f"{time.time() - start} seconds to run learning fair representation")
        
        start = time.time()
        self.__reweighing(train_true_labels, validation_to_predict)
        print(f"{time.time() - start} seconds to run reweighing")
        
        start = time.time()
        self.__reject_option_classification(train_true_labels, train_predictions, validation_to_predict)
        print(f"{time.time() - start} seconds to run reject option classification")
        
        start = time.time()
        self.__calibrated_equal_odds(train_true_labels, train_predictions, validation_to_predict)
        print(f"{time.time() - start} seconds to run calibrated equal odds")
        
        start = time.time()
        self.__equal_odds(train_true_labels, train_predictions, validation_to_predict)
        print(f"{time.time() - start} seconds to run equal odds")
    
    
    def __get_aif360_datasets(self, x_train, x_validation, training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities):
        # To avoid recomputing for each debiasing methodology we create the aif360.datasets and pass them to each function
        # aif360.datasets take a dataframe argument, so make sure this is the type for both x_train and x_validation
        if isinstance(x_train, pd.DataFrame):
            df_train = x_train
            df_validation = x_validation
        elif isinstance(x_train, np.ndarray):
            if sparse.issparse(x_train):
                df_train = pd.DataFrame.sparse.from_spmatrix(x_train)
                df_validation = pd.DataFrame.sparse.from_spmatrix(x_validation)
            else:
                df_train = pd.DataFrame(x_train)
                df_validation = pd.DataFrame(x_validation)
        else:
            raise RuntimeError("Pre-processing results in an unrecognised datatype. Please make sure runing pre-processing returns a pandas dataframe or a numpy array. If you have used no pre-processing, make sure your data is of the expected type")
        
        # Finish creating the datasets for training now that we know df_train is a pandas dataframe     
        df_train["Is Privileged"] = self.privilege_train
        
        df_train_true_labels = df_train.copy()
        df_train_true_labels["Class Label"] = self.df_y_train.to_numpy()
        
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
    
    
    def show_fairea_graph(self, error_metric: str, fairness_metric: str) -> None:
        data = self.__create_scatter_with_fairea(error_metric, fairness_metric)
        
        layout = go.Layout(
            xaxis_title=f"Bias ({fairness_metric})", 
            yaxis_title=f"Error ({error_metric})"
        )
        
        figure = go.Figure(data=data, layout=layout)
        figure.update_layout(title=f"Debias Methodologies impact on {error_metric} and {fairness_metric}")
        figure.update_xaxes(tick0=0, dtick=0.1) 
        figure.update_yaxes(tick0=0, dtick=0.1)
        figure.show() 
    
    
    def show_many_fairea_graphs(self, error_metrics: list[str], fairness_metrics: list[str]) -> None:
        if isinstance(error_metrics, str):
            error_metrics = [error_metrics]
        if isinstance(fairness_metrics, str):
            fairness_metrics = [fairness_metrics]
            
        subplots = make_subplots(len(error_metrics), len(fairness_metrics))
        
        for row, error_metric in enumerate(error_metrics):
            for col, fairness_metric in enumerate(fairness_metrics):
                plots = self.__create_scatter_with_fairea(error_metric, fairness_metric, showlegend=(row==0 and col==0))
                for plot in plots:
                    # The subplots are 1 indexed, while enumerate is 0 index, hence the add 1
                    subplots.add_trace(plot, row=row+1, col=col+1)
                
                subplots.update_xaxes(tick0=0, dtick=0.1, row=row+1, col=col+1) 
                subplots.update_yaxes(tick0=0, dtick=0.1, row=row+1, col=col+1)
                
                if col == 0:
                    subplots.update_yaxes(title_text=f"Error ({error_metric})", col=col+1, row=row+1)
                if row == len(error_metrics) - 1:
                    subplots.update_xaxes(title_text=f"Bias ({fairness_metric})", col=col+1, row=row+1)
        
        subplots.update_layout(height=800*len(fairness_metrics), width=200*len(error_metrics)) 
        subplots.show()
                    
            
    def show_all_fairea_graphs(self) -> None:
        self.show_many_fairea_graphs(self.get_error_metric_names(), self.get_bias_metric_names()) 
    
    
    def __create_debias_methodology_scatter(self, debias_methodology: str, error_metric: str, fairness_metric: str, color: str, showlegend=True) -> go.Scatter:
        # Statistics regarding the debiasing technique to be plotted as a single point
        debias_x = self.__metrics_by_debiasing_technique[debias_methodology]["fairness"][fairness_metric]["value"]
        debias_y = self.__metrics_by_debiasing_technique[debias_methodology]["error"][error_metric]["value"]
        
        # Error bars to put on the single point
        confidence_interval = self.__metrics_by_debiasing_technique[debias_methodology]["fairness"][fairness_metric]["confidence interval"]
        error_bars_x = (confidence_interval[0] - confidence_interval[1]) / 2
        
        confidence_interval = self.__metrics_by_debiasing_technique[debias_methodology]["error"][error_metric]["confidence interval"]
        error_bars_y = (confidence_interval[0] - confidence_interval[1]) / 2
        
        debias_result = go.Scatter(
            x=[debias_x], 
            error_x={"type": "data", "array": [error_bars_x]}, 
            y=[debias_y], 
            error_y={"type": "data", "array": [error_bars_y]}, 
            mode="markers", 
            name=f"{debias_methodology} outcome",
            showlegend=showlegend,
            marker_color=color
        )
        
        return debias_result
    
    
    def __create_scatter_with_fairea(self, error_metric: str, fairness_metric: str, debias_methodologies: list[str]=None, showlegend=True) -> list[go.Scatter]:
        if debias_methodologies is None:
            # If the debias_methodologies to be plotted are not specified, plot all but 'no debiasing' which will be covered by fairea
            debias_methodologies = self.get_debias_methodologies()
            debias_methodologies.remove("no debiasing")
            
        fairea_labels = []
        fairea_x = []
        fairea_y = []
        
        # Create the fairea curve 
        for mutation, metric in self.__fairea:
            fairea_labels.append(f"F_{int(mutation * 100)}")
            fairea_x.append(metric["fairness"][fairness_metric])
            fairea_y.append(metric["error"][error_metric])
        
        
        colors = px.colors.qualitative.Dark24
        
        fairea_curve = go.Scatter(
            x=fairea_x, 
            y=fairea_y, 
            mode="lines+markers+text", 
            name="fairea baseline", 
            text=fairea_labels, 
            textposition="bottom right", 
            showlegend=showlegend,
            line_color=colors[0]
        )
        colors = colors[1:]
        
        scatter_plots = [self.__create_debias_methodology_scatter(method, error_metric, fairness_metric, color=color, showlegend=showlegend) for method, color in zip(debias_methodologies, colors)]
        return [*scatter_plots, fairea_curve]
    
    
    def get_debias_methodologies(self) -> list[str]:
        return list(self.__metrics_by_debiasing_technique.keys())
    
    
    def get_error_metric_names(self) -> list[str]:
        some_debias = list(self.__metrics_by_debiasing_technique.keys())[0]
        return list(self.__metrics_by_debiasing_technique[some_debias]["error"].keys())
            
                 
    def get_bias_metric_names(self) -> list[str]:
        some_debias = list(self.__metrics_by_debiasing_technique.keys())[0]
        return list(self.__metrics_by_debiasing_technique[some_debias]["fairness"].keys())
    
    
    def get_raw_data(self) -> list[str]:
        return self.__metrics_by_debiasing_technique             
    
    
    def __no_debiasing(self, x_train, x_validation):
        self.model.fit(x_train, self.df_y_train)
        
        training_predicted_values = self.model.predict(x_train)
        training_probabilities = self.model.predict_proba(x_train)[:, 1]
                   
        validation_predicted_values = self.model.predict(x_validation)
        validation_probabilities = self.model.predict_proba(x_validation)[:, 1]
        
        self.__metrics_by_debiasing_technique["no debiasing"] = bootstrap_all_metrics(self.df_y_validation.to_numpy(), validation_predicted_values, self.privilege_validation)
        
        # Most debiasing analysis functions won't return anything, however having default results is needed for fairea and postprocessing debiasing 
        return training_predicted_values, training_probabilities, validation_predicted_values, validation_probabilities
        
    # def __optimized_preprocessing(self):
        # this looks like it requires more knowledge of the dataset than I expect this framework to know at this point in time, so leaving this unimplemented
        # Refer back to these when it comes time to implement
        # from aif360.algorithms.preprocessing import OptimPreproc
        # from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
        pass
    
    
    def __learning_fair_representation(self, train_true_labels, validation_to_predict):
        # TODO
        # This method can product wildly different results. I should investigate how to make it more consistent.
        # More specifics:
        # This debiasing method involves creating a mapping from the input space to an intermediate representation, called a prototype, which is then what the ml model uses to classify. Finding this map from input to prototype involves solving an optimisation problem. This is done numerically, with randomly chosen initial values. For some values, it appears to produce absolutely terrible results
        
        true_values = self.df_y_validation.to_numpy()
        
        # Run the debiasing method with the given number of prototypes so we can start to understand how the parameter changes debiasing and what a good value might be. 
        for number_of_prototypes in [5, 10, 15]:
            print(f"fair representation with k = {number_of_prototypes}")

            training_dataset = train_true_labels.copy()

            # Applying learning fair representation to the training data
            fair_representation = LFR(unprivileged_groups=[{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}], k=number_of_prototypes)
            
            transformed_data = fair_representation.fit_transform(training_dataset)
            
            # This debiasing methodology also mutates the labels. If there is a single label,the classifier may raise an exception, so this case must be handled seprately
            classes = np.unique(transformed_data.labels.ravel())
            if len(classes) == 1:
                # It is reasonable to assume that if only one class exists in the training data, it will be the only predicted value.
                predicted_values = np.full(len(self.df_y_validation), classes[0])
            else:
                # Note that this debiasing methodology also seems to update the class labels. I'm not entirely clear on why, but does seem to get better results with the updated labels.
                self.model.fit(transformed_data.features, transformed_data.labels.ravel())

                validation_dataset = validation_to_predict.copy()
                validation_dataset = fair_representation.transform(validation_dataset)
                predicted_values = self.model.predict(validation_dataset.features)
                
            

            self.__metrics_by_debiasing_technique[f"learning fair representation with {number_of_prototypes} prototypes"] = bootstrap_all_metrics(true_values, predicted_values, self.privilege_validation)


    def __reweighing(self, train_true_labels, validation_to_predict):
        training_dataset = train_true_labels.copy()
        
        # Applying reweighing to the training data
        reweighing = Reweighing(unprivileged_groups=[{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}])
        transformed_data = reweighing.fit_transform(training_dataset)
        self.model.fit(transformed_data.features, transformed_data.labels.ravel(), sample_weight=transformed_data.instance_weights)
        
        # Applying the required modifications to the validation data and getting results for metric calculation
        # Note that we don't need to apply reweighing because that only impacts the training stage
        predicted_values = self.model.predict(validation_to_predict.copy().features)
        true_values = self.df_y_validation.to_numpy()
        
        self.__metrics_by_debiasing_technique["reweighing"] = bootstrap_all_metrics(true_values, predicted_values, self.privilege_validation)
        
    
    # def __disparate_impact_remover(self):
        # Notes on ways we can get more out of each debiasing method, for those that have additional parameters we can set
        # Adjust repair_level. Default is 1 (full), which the paper points out is likely to degrade classification accuracy. 
        # Could include some partial values, say 0.5 and 0.75 to get a better idea how adjusting this may impact model performance?
        
        # Note: In the code for the paper from which this class largely sourced, the DisparateImpactRemover was applied to both the train and validation data. I don't really understand if this is correct, since 

        # Not implemented yet since it might require removal of the protected fields, which is not something I can do at the moment
        # from aif360.algorithms.preprocessing import DisparateImpactRemover
        pass
    
    
    # def __prejudice_remover(self):
        # This implementation does not work with the framework and trying to figure out what it is actually doing is confusing
        # Honestly, I find some of the code suspect. Remote execution of a python file rather than importing it
        # from aif360.algorithms.inprocessing.prejudice_remover import PrejudiceRemover
        pass
    
    
    def __reject_option_classification(self, train_true_labels, train_predictions, validation_predicted_values):
        # We want results for optimising with respect to each of the bias metrics, as it is unclear what the tradeoff will be
        for metric_name in ["Statistical parity difference", "Average odds difference", "Equal opportunity difference"]:
            training_dataset_true = train_true_labels.copy()
            training_dataset_predictions = train_predictions.copy()

            reject_option_classification = RejectOptionClassification([{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}], metric_name=metric_name)
            reject_option_classification.fit(training_dataset_true, training_dataset_predictions)
            
            validation_dataset = validation_predicted_values.copy()
            predicted_values = reject_option_classification.predict(validation_dataset).labels.ravel()
            true_values = self.df_y_validation.to_numpy()

            self.__metrics_by_debiasing_technique[f"reject option classification {metric_name.lower()} optimised"] = bootstrap_all_metrics(true_values, predicted_values, self.privilege_validation)        


    def __calibrated_equal_odds(self, train_true_labels, train_predictions, validation_predicted_values):
        for cost_contraint in ["fpr", "fnr", "weighted"]:

            training_dataset_true = train_true_labels.copy()
            training_dataset_predictions = train_predictions.copy()
            
            calibrated_equal_odds = CalibratedEqOddsPostprocessing([{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}], cost_constraint=cost_contraint)
            calibrated_equal_odds.fit(training_dataset_true, training_dataset_predictions)
            
            validation_dataset = validation_predicted_values.copy()
            predicted_values = calibrated_equal_odds.predict(validation_dataset).labels.ravel()
            true_values = self.df_y_validation.to_numpy()
            
            self.__metrics_by_debiasing_technique[f"calibrated equal odds using {cost_contraint} cost"] = bootstrap_all_metrics(true_values, predicted_values, self.privilege_validation)          


    def __equal_odds(self, train_true_labels, train_predictions, validation_predicted_values):

        training_dataset_true = train_true_labels.copy()
        training_dataset_predictions = train_predictions.copy()
        
        calibrated_equal_odds = EqOddsPostprocessing([{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}])
        calibrated_equal_odds.fit(training_dataset_true, training_dataset_predictions)
        
        validation_dataset = validation_predicted_values.copy()
        predicted_values = calibrated_equal_odds.predict(validation_dataset).labels.ravel()
        true_values = self.df_y_validation.to_numpy()
        
        self.__metrics_by_debiasing_technique["equal odds"] = bootstrap_all_metrics(true_values, predicted_values, self.privilege_validation)