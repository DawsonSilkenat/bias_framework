import pandas as pd
import numpy as np
from ml_dummy import ML_Model
from metrics import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing

# This is only for information on runtime rather than used functionally
import time

# aif360 seems to do something pandas doesn't like, and it makes it hard to debug when all I read are these warning messages
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



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
        
        self.__fairea = None
        self.__metrics_by_debiasing_technique = dict()
    
    
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
            The first dictionary indicates that if sex has value 1 and age has value 1 then the individual belongs to the privileged group
            The second dictionary indicates that if sex has value 0 then the individual belongs to the privileged group (regardless of age)
        """
        self.privilege_function = lambda x: int(
            # Checks if a row matches any of the valid groupings
            any([
                # Checks if all the requirements for a particular grouping are met
                all([x[name] == value for name, value in grouping]) for grouping in privileged_combinations])
        )
    
    
    def set_unprivileged_combinations(self, unprivileged_combinations: list[dict[str: any]]) -> None:
        """Update the function which determines if an element belongs to the privileged or unprivileged class by specifying which groupings are not considered privileged

        Args:
            unprivileged_combinations: a list of dictionaries, each dictionary maps a set of columns name to the values they must take. Note: Each columns name must map to a single value, not a list of possible values. The values of any column names not included in the dictionary are ignored
            
            Example from aif360 documentation:
            [{'sex': 1, 'age': 1}, {'sex': 0}]
            The first dictionary indicates that if sex has value 1 and age has value 1 then the individual belongs to the unprivileged group
            The second dictionary indicates that if sex has value 0 then the individual belongs to the unprivileged group (regardless of age)
        """
        self.privilege_function = lambda x: int(
            # Checks if a row matches any of the valid groupings, and negates it so we are unprivileged
            not any([
                # Checks if all the requirements for a particular grouping are met
                all([x[name] == value for name, value in grouping]) for grouping in unprivileged_combinations])
        )
    
    
    def run_framework(self):
        start = time.time()
        fairea_args = self.__no_debiasing()
        print(f"{time.time() - start} seconds to run with no debiasing")
        start = time.time()
        self.__fairea = fairea_model_mutation(*fairea_args)
        print(f"{time.time() - start} seconds to get fairea baseline")
        start = time.time()
        self.__reweighing()
        print(f"{time.time() - start} seconds to run reweighing")
    
    
    def show_fairea_graph(self, error_metric: str, fairness_metric: str) -> None:
        figure = self.__create_figure_with_fairea(error_metric, fairness_metric)
        figure.update_layout(title=f"Debias Methodologies impact on {error_metric} and {fairness_metric}")
        figure.show() 
    
    
    def show_many_fairea_graphs(self, error_metric: list[str], fairness_metric: list[str]) -> None:
        if isinstance(error_metric, str):
            error_metric = [error_metric]
        if isinstance(fairness_metric, str):
            fairness_metric = [fairness_metric]
            
        make_subplots
    
    
    def show_all_fairea_graphs(self) -> None:
        self.show_many_fairea_graphs(self.get_error_metric_names(), self.get_bias_metric_names()) 
    
    
    def __create_debias_methodology_scatter(self, debias_methodology: str, error_metric: str, fairness_metric: str) -> go.Scatter:
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
            error_x=dict(type='data', array=[error_bars_x]), 
            y=[debias_y], 
            error_y=dict(type='data', array=[error_bars_y]), 
            mode="markers", 
            name=f"{debias_methodology} outcome"
        )
        
        return debias_result
    
    
    def __create_figure_with_fairea(self, error_metric: str, fairness_metric: str, debias_methodologies: list[str]=None) -> go.Figure:
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
        
        fairea_curve = go.Scatter(
            x=fairea_x, 
            y=fairea_y, 
            mode="lines+markers+text", 
            name="fairea baseline", 
            text=fairea_labels, 
            textposition="bottom right"
        )
        
        layout = go.Layout(
            xaxis_title=f"Bias ({fairness_metric})", 
            yaxis_title=f"Error ({error_metric})",
            # width=800,
            # height=800,
        )
        
        scatter_plots = [self.__create_debias_methodology_scatter(method, error_metric, fairness_metric) for method in debias_methodologies]
        figure = go.Figure(data=[*scatter_plots, fairea_curve], layout=layout)
        figure.update_xaxes(tick0=0, dtick=0.1) 
        figure.update_yaxes(tick0=0, dtick=0.1)
        
        return figure
    
    
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
    
    
    def __no_debiasing(self):
        self.model.fit(self.df_x_train, self.df_y_train)
        predicted_values = self.model.predict(self.df_x_validate)
        true_values = self.df_y_validate.to_numpy()
        
        self.__metrics_by_debiasing_technique["no debiasing"] = bootstrap_error_metrics(self.df_x_validate, true_values, predicted_values, self.privilege_function)
        
        # Most debiasing analysis functions won't return anything, this one does so we can run fairea without recomputing
        return self.df_x_validate, true_values, predicted_values, self.privilege_function
        
        
    def __reweighing(self):
        # Creating a StandardDataset with the required information about privileged group
        df_train = self.df_x_train.copy()
        df_train["Is Privileged"] = df_train.apply(self.privilege_function, axis=1)
        df_train["Class Label"] = self.df_y_train.to_numpy()
        
        training_dataset = StandardDataset(df_train, 
                    label_name="Class Label", 
                    favorable_classes=[1],
                    protected_attribute_names=["Is Privileged"], 
                    privileged_classes=[[1]]
                    )
        
        # Applying reweighing to the training data
        reweighing = Reweighing(unprivileged_groups=[{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}])
        transformed_data = reweighing.fit_transform(training_dataset)
        self.model.fit(transformed_data.features, transformed_data.labels.ravel(), sample_weight=transformed_data.instance_weights)
        
        # Applying the required modifications to the validation data and getting results for metric calculation
        df_x_validate = self.df_x_validate.copy()
        df_x_validate["Is Privileged"] = df_x_validate.apply(self.privilege_function, axis=1)
        predicted_values = self.model.predict(df_x_validate)
        true_values = self.df_y_validate.to_numpy()
        
        self.__metrics_by_debiasing_technique["reweighing"] = bootstrap_error_metrics(self.df_x_validate, true_values, predicted_values, self.privilege_function)
        
