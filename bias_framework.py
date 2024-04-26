import pandas as pd
import numpy as np
from ml_dummy import ML_Model
from metrics import *
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import LFR, Reweighing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from sklearn.calibration import CalibratedClassifierCV

# I quite like the idea of Discrimination aware Ensemble, but it doesn't work with arbitrary classifiers. Might be something to look into later

# This is only for information on runtime rather than used functionally
import time

# aif360 seems to do something pandas doesn't like, and it makes it hard to debug when all I read are these warning messages
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



class Bias_Framework:
    def __init__(self, model: ML_Model, df_training_data: pd.DataFrame, df_validation_data: pd.DataFrame) -> None:
        """Creates an instance of the bias framework applied to the specified model and data

        Args:
            model: The ML model to which the bias framework will be applied. This model must have a fit method and predict method. I am assuming at the moment that this will be an sklearn ml model, might try to modify this to be more flexable later.
            training_data: The data for training the ML model. It is assumed that the last column is the target variable.
            validation_data: The data for which fairness metrics. It is assumed that the columns are the same as training_data.
        """
        if not df_training_data.columns.equals(df_validation_data.columns):
            raise ValueError("The training and validation dataframes must contain the same columns")
        
        # Not all models implement a function for prediction probability. Probabilities are required for some postprocessing debiasing. The CalibratedClassifierCV method preforms probability calibration, which also adds this functionality if it is not part of the model
        self.model = CalibratedClassifierCV(estimator=model)
        # self.model = model
        
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
        validation_true_values, validation_predicted_values = self.__no_debiasing()
        print(f"{time.time() - start} seconds to run with no debiasing")
        
        start = time.time()
        self.__fairea = fairea_model_mutation(self.df_x_validate, validation_true_values, validation_predicted_values, self.privilege_function)
        print(f"{time.time() - start} seconds to get fairea baseline")
        
        # TODO uncomment, I just want to reduce runtime while testing
        start = time.time()
        self.__learning_fair_representation() 
        print(f"{time.time() - start} seconds to run learning fair representation")
        
        start = time.time()
        self.__reweighing()
        print(f"{time.time() - start} seconds to run reweighing")
    
    
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
    
    
    def __no_debiasing(self):
        self.model.fit(self.df_x_train, self.df_y_train)
        validation_predicted_values = self.model.predict(self.df_x_validate)
        validation_true_values = self.df_y_validate.to_numpy()
        
        self.__metrics_by_debiasing_technique["no debiasing"] = bootstrap_error_metrics(self.df_x_validate, validation_true_values, validation_predicted_values, self.privilege_function)
        
        # Most debiasing analysis functions won't return anything, however having default results is needed for fairea and postprocessing debiasing 
        return validation_true_values, validation_predicted_values
        
    def __0ptimized_preprocessing(self):
        # TODO this looks like it requires more knowledge of the dataset than I expect this framework to know at this point in time, so leaving this unimplemented
        # Refer back to these when it comes time to implement
        # from aif360.algorithms.preprocessing import OptimPreproc
        # from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
        pass
    
    
    def __learning_fair_representation(self):
        # TODO
        # This method can product wildly different results. I should investigate how to make it more consistent.
        # More specifics:
        # This debiasing method involves creating a mapping from the input space to an intermediate representation, called a prototype, which is then what the ml model uses to classify. Finding this map from input to prototype involves solving an optimisation problem. This is done numerically, with randomly chosen initial values. For some values, it appears to produce absolutely terrible results

        
        # TODO 
        # Notes on ways we can get more out of each debiasing method, for those that have additional parameters we can set
        # Adjust k, the number of intermediary prototypes before classification
        # Ax, Ay, Az, not super clear on how I would go about modifying these
        
        # Creating a StandardDataset with the required information about privileged group
        df_train = self.df_x_train.copy()
        df_train["Is Privileged"] = df_train.apply(self.privilege_function, axis=1)
        df_train["Class Label"] = self.df_y_train.to_numpy()
        
        training_dataset = StandardDataset(
            df_train, 
            label_name="Class Label", 
            favorable_classes=[1],
            protected_attribute_names=["Is Privileged"], 
            privileged_classes=[[1]]
        )

        # Applying learning fair representation to the training data
        fair_representation = LFR(unprivileged_groups=[{"Is Privileged" : 0}], privileged_groups=[{"Is Privileged" : 1}])
        transformed_data = fair_representation.fit_transform(training_dataset)
        self.model.fit(transformed_data.features, transformed_data.labels.ravel())
        
        # Applying the required modifications to the validation data and getting results for metric calculation
        df_x_validate = self.df_x_validate.copy()
        df_x_validate["Is Privileged"] = df_x_validate.apply(self.privilege_function, axis=1)
        
        
        # We need a dataset, rather than a dataframe, to apply our fair_representation to. 
        # The class label shouldn't matter here but is a required argument for StandardDataset.
        df_x_validate["Class Label"] = np.zeros(len(df_x_validate))
        validation_dataset = StandardDataset(
            df_x_validate, 
            label_name="Class Label", 
            favorable_classes=[1],
            protected_attribute_names=["Is Privileged"], 
            privileged_classes=[[1]]
        )
        
        validation_dataset = fair_representation.transform(validation_dataset)
        
        predicted_values = self.model.predict(validation_dataset.features)
        true_values = self.df_y_validate.to_numpy()
        
        self.__metrics_by_debiasing_technique["learning fair representation"] = bootstrap_error_metrics(self.df_x_validate, true_values, predicted_values, self.privilege_function)
    
        
    def __reweighing(self):
        # Creating a StandardDataset with the required information about privileged group
        df_train = self.df_x_train.copy()
        df_train["Is Privileged"] = df_train.apply(self.privilege_function, axis=1)
        df_train["Class Label"] = self.df_y_train.to_numpy()
        
        training_dataset = StandardDataset(
            df_train, 
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
        # Note that we don't need to apply reweighing because that only impacts the training stage
        df_x_validate = self.df_x_validate.copy()
        df_x_validate["Is Privileged"] = df_x_validate.apply(self.privilege_function, axis=1)
        predicted_values = self.model.predict(df_x_validate)
        true_values = self.df_y_validate.to_numpy()
        
        self.__metrics_by_debiasing_technique["reweighing"] = bootstrap_error_metrics(self.df_x_validate, true_values, predicted_values, self.privilege_function)
        
    
    def __disparate_impact_remover(self):
        # TODO 
        # Notes on ways we can get more out of each debiasing method, for those that have additional parameters we can set
        # Adjust repair_level. Default is 1 (full), which the paper points out is likely to degrade classification accuracy. 
        # Could include some partial values, say 0.5 and 0.75 to get a better idea how adjusting this may impact model performance?
        
        # Note: In the code for the paper from which this class largely sourced, the DisparateImpactRemover was applied to both the train and validation data. I don't really understand if this is correct, since 

        # Not implemented yet since it might require removal of the protected fields, which is not something I can do at the moment
        # from aif360.algorithms.preprocessing import DisparateImpactRemover
        pass
    
    
    def __prejudice_remover(self):
        # This implementation does not work with the framework and trying to figure out what it is actually doing is confusing
        # Honestly, I find some of the code suspect. Remote execution of a python file rather than importing it
        # from aif360.algorithms.inprocessing.prejudice_remover import PrejudiceRemover
        pass
    
    
    def __reject_option_classification(self):
        pass
        
    
    
        
