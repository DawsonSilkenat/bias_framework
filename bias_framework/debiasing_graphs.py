import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from .baselines import Baseline


class DebiasingGraphsObject:
    """This class is responsible for producing graphs related to debiasing. 
    To do this, it stores the recorded metrics in a nested dictionary with format:
        debiasing technique name -> ['error', 'fairness'] -> metric name -> metric values.
    A baseline curve to compare the effects of the debiasing against is also recorded using a purpose build class found
     the baselines package.
    Optionally, a name can be provided. This is most useful when adding two graphs together, so you can identify which
     results originated in each graph.
    """
    
    def __init__(self, metrics_by_debiasing_technique: dict[str, dict[str, dict[str, float]]],
                 baseline_curve: Baseline | list[Baseline], name: str = None) -> None:
        if not isinstance(baseline_curve, list):
            baseline_curve = [baseline_curve]
            
        self.baseline_curve = baseline_curve
        self.metrics_by_debiasing_technique = metrics_by_debiasing_technique 
        self.name = name


    def get_debias_methodologies(self) -> list[str]:
        """Returns: list[str]: The list of debiasing methodologies for which this class has recorded metrics"""
        return list(self.metrics_by_debiasing_technique.keys())
    
    
    def get_error_metric_names(self) -> list[str]:
        """Returns: list[str]: The list of error metrics this class has recorded for the debiasing methodologies
        Useful for identifying the correct keywords to specify a graph
        """
        some_debias_method = list(self.metrics_by_debiasing_technique.keys())[0]
        return list(self.metrics_by_debiasing_technique[some_debias_method]["error"].keys())
            
                 
    def get_bias_metric_names(self) -> list[str]:
        """Returns: The list of bias metrics this class has recorded for the debiasing methodologies
        Useful for identifying the correct keywords to specify a graph
        """
        some_debias_method = list(self.metrics_by_debiasing_technique.keys())[0]
        return list(self.metrics_by_debiasing_technique[some_debias_method]["fairness"].keys())
    
    
    def get_raw_data(self) -> dict[str, dict[str, dict[str, float]]]:
        """Returns: The full dictionary of recorded metrics by debiasing techniques
        """
        return self.metrics_by_debiasing_technique  
    
    
    def set_name(self, name: str) -> None:
        """Updates the recorded name, important for combining graphs so you can identify the origin of the plots"""
        self.name = name 
        if len(self.baseline_curve) == 1:
            self.baseline_curve.name = name
        
    
    def get_single_graph(self, error_metric: str, fairness_metric: str) -> go.Figure:
        """Returns the plotly figure of a graph showing the impact of debiasing as measured using the specified metrics

        Args:
            error_metric (str): which error metric to use on the plot, call get_error_metric_names to find available values
            fairness_metric (str): which bias metric to use on the plot, call get_bias_metric_names to find available values

        Returns:
            go.Figure: The requested plot as a plotly figure
        """
        data = self.__create_scatter_with_baseline(error_metric, fairness_metric)
        
        layout = go.Layout(
            xaxis_title=f"Bias ({fairness_metric})", 
            yaxis_title=f"Error ({error_metric})"
        )
        
        figure = go.Figure(data=data, layout=layout)
        figure.update_layout(title=f"Debias Methodologies impact on {error_metric} and {fairness_metric}")
        figure.update_xaxes(tick0=0, dtick=0.1) 
        figure.update_yaxes(tick0=0, dtick=0.1)
        return figure
    
    
    def show_single_graph(self, error_metric: str, fairness_metric: str) -> None:
        """Displays a graph showing the impact of debiasing as measured using the specified metrics
        
        Args:
            error_metric (str): which error metric to use on the plot, call get_error_metric_names to find available values
            fairness_metric (str): which bias metric to use on the plot, call get_bias_metric_names to find available values
        """
        self.get_single_graph(error_metric, fairness_metric).show()
    
        
    def get_subplots(self, error_metrics: list[str], fairness_metrics: list[str]) -> go.Figure:
        """Returns the plotly figure of a matrix of graphs showing the impact of debiasing as measured using the specified metrics

        Args:
            error_metrics (list[str]): which error metrics to use on the plot, call get_error_metric_names to find available values
            fairness_metrics (list[str]): which bias metrics to use on the plot, call get_bias_metric_names to find available values

        Returns:
            go.Figure: The requested plot as a plotly figure
        """
        if isinstance(error_metrics, str):
            error_metrics = [error_metrics]
        if isinstance(fairness_metrics, str):
            fairness_metrics = [fairness_metrics]
            
        subplots = make_subplots(len(error_metrics), len(fairness_metrics))
        
        for row, error_metric in enumerate(error_metrics):
            for col, fairness_metric in enumerate(fairness_metrics):
                plots = self.__create_scatter_with_baseline(error_metric, fairness_metric, showlegend=(row==0 and col==0))
                for plot in plots:
                    # The subplots are 1 indexed, while enumerate is 0 index, hence the add 1
                    subplots.add_trace(plot, row=row+1, col=col+1)
                
                # Standardise the spacing of ticks across plots to make easier to interpret at a glance
                subplots.update_xaxes(tick0=0, dtick=0.1, row=row+1, col=col+1)
                subplots.update_yaxes(tick0=0, dtick=0.1, row=row+1, col=col+1)
                
                if col == 0:
                    subplots.update_yaxes(title_text=f"Performance ({error_metric})", col=col+1, row=row+1)
                if row == len(error_metrics) - 1:
                    subplots.update_xaxes(title_text=f"Bias ({fairness_metric})", col=col+1, row=row+1)
        
        subplots.update_layout(height=800*len(fairness_metrics), width=200*len(error_metrics)) 
        return subplots
        
    
    def show_subplots(self, error_metrics: list[str], fairness_metrics: list[str]) -> None:
        """Displays a matrix of graphs showing the impact of debiasing as measured using the specified metrics
        
        Args:
            error_metrics (list[str]): which error metrics to use on the plot, call get_error_metric_names to find available values
            fairness_metrics (list[str]): which bias metrics to use on the plot, call get_bias_metric_names to find available values
        """
        self.get_subplots(error_metrics, fairness_metrics).show()
    
     
    def get_all_subplots(self) -> go.Figure:
        """Returns the plotly figure of a matrix of graphs showing the impact of debiasing as measured using all available metrics"""
        return self.get_subplots(self.get_error_metric_names(), self.get_bias_metric_names()) 
      
        
    def show_all_subplots(self) -> None:
        """Displays a matrix of graphs showing the impact of debiasing as measured using all available metrics"""
        self.get_all_subplots().show()
        
    
    def __create_debias_methodology_point(self, debias_methodology: str, error_metric: str, fairness_metric: str,
                                          color: str, showlegend=True) -> go.Scatter:
        """Returns a plotly scatterplot object for a single point: the results of debias_methodology as measured using
        error_metric and fairness_metric

        Args:
            debias_methodology (str): Which of the available debias methodology this point represents
            error_metric (str): Which metric is used to to measure error
            fairness_metric (str): Which metric is used to to measure bias
            color (str): Which color should be used to represent this point on a graph derived from the returned object
            showlegend (bool, optional): If this point should be included on a legend in a resulting graph. Useful for
                subplots so that you don't end up with a repeating legend. Defaults to True.

        Returns:
            go.Scatter: The point to be placed on a graph
        """
        
        # Statistics regarding the debiasing technique to be plotted as a single point
        debias_x = self.metrics_by_debiasing_technique[debias_methodology]["fairness"][fairness_metric]["value"]
        debias_y = self.metrics_by_debiasing_technique[debias_methodology]["error"][error_metric]["value"]
        
        # Error bars to put on the single point
        confidence_interval = self.metrics_by_debiasing_technique[debias_methodology]["fairness"][fairness_metric]["confidence interval"]
        error_bars_x = (confidence_interval[0] - confidence_interval[1]) / 2
        
        confidence_interval = self.metrics_by_debiasing_technique[debias_methodology]["error"][error_metric]["confidence interval"]
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
    
    
    def __create_scatter_with_baseline(self, error_metric: str, fairness_metric: str, debias_methodologies: list[str]=None,
                                       showlegend=True) -> list[go.Scatter]:
        """Create a scatterplot using the specified arguments, to be plotted on either an individual plot or with subplots

        Args:
            error_metric (str): Which metric is used to to measure error
            fairness_metric (str): Which metric is used to to measure bias
            debias_methodologies (list[str], optional): Which debias methodologies should be included on the resulting plot.
                Defaults to all available in the dictionary except "no debiasing".
            showlegend (bool, optional): If this plot should be included on a legend in a resulting graph.
                Useful for subplots so that you don't end up with a repeating legend. Defaults to True.

        Returns:
            list[go.Scatter]: The list of components to be added to a plotly figure or subplot in order to produce the described graph.
        """
        
        # Need a large number of colors because we want to be able to plot the composition of graphs
        colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
        
        if debias_methodologies is None:
            # If the debias_methodologies to be plotted are not specified, plot all but 'no debiasing' 
            debias_methodologies = self.get_debias_methodologies()
            debias_methodologies.remove("no debiasing")
        elif isinstance(debias_methodologies, str):
            debias_methodologies = [debias_methodologies]

        baseline_curves = []
        for baseline_curve in self.baseline_curve:
            baseline_curves.append(baseline_curve.get_baseline_curve(error_metric, fairness_metric, colors[0], showlegend))
            colors = colors[1:]
        
        scatter_plots = [self.__create_debias_methodology_point(method, error_metric, fairness_metric, color=color, showlegend=showlegend)
                         for method, color in zip(debias_methodologies, colors)]
        return [*scatter_plots, *baseline_curves]
    
    
    def __add__(self, other) -> "DebiasingGraphsComposition":
        if isinstance(other, DebiasingGraphsObject):
            return DebiasingGraphsComposition(self, other)
        return NotImplemented
    
    
class DebiasingGraphsComposition:
    """Wrapper around DebiasingGraphsObject to make it easier to combine plots without losing information about the origin of the information
    """
    
    def __init__(self, *debiasing_graphs: DebiasingGraphsObject) -> None:
        self.debiasing_graphs = debiasing_graphs
        self.composite_debias_graph = None
    
    
    def __add__(self, other):
        if isinstance(other, DebiasingGraphsComposition): 
            return DebiasingGraphsComposition(*self.debiasing_graphs, *other.debiasing_graphs)
        if isinstance(other, DebiasingGraphsObject):
            return DebiasingGraphsComposition(*self.debiasing_graphs, other)
        return NotImplemented
        
    
    def __compute_composite_debiasing_graph(self) -> None:
        metrics_by_debiasing_technique = dict()
        baselines = []
        
        for graph_object in self.debiasing_graphs:
            name = graph_object.name
            data = graph_object.get_raw_data()
            
            # Combine the dictionaries of recorded metrics, updating the keys to ensure that names do not overlap
            for key, value in data.items():
                if name:
                    key = name + " " + key
                
                i = 1
                updated_key = key
                while updated_key in metrics_by_debiasing_technique.keys():
                    updated_key = key + " " + str(i)
                    i += 1
                
                metrics_by_debiasing_technique[updated_key] = value
            
            baselines.extend(graph_object.baseline_curve)
            
        self.composite_debias_graph = DebiasingGraphsObject(metrics_by_debiasing_technique, baselines)
    
    
    def __getattr__(self, attr):
        # Handles method calls intended for self.composite_debias_graph
        if self.composite_debias_graph == None:
            self.__compute_composite_debiasing_graph()
            if hasattr(self.composite_debias_graph, attr):
                return getattr(self.composite_debias_graph, attr)
        
        raise AttributeError(f"No attribute '{attr}'")

    
    