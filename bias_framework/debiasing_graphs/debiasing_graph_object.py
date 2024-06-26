import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from ..baselines import Baseline


class DebiasingGraphsObject:
    """This class is responsible for producing graphs related to 
    debiasing. 
    To do this, it stores the recorded metrics in a nested dictionary 
    with format: debiasing technique name -> ['error', 'fairness'] 
    -> metric name -> metric values.
    A baseline curve to compare the effects of the debiasing against is 
    also recorded using a purpose build class found the the baselines 
    package.
    Optionally, a name can be provided. This is most useful when adding 
    two graphs together, so you can identify which results originated in 
    each graph.
    """

    def __init__(
            self, 
            metrics_by_debiasing_technique: dict[
                str, dict[
                    str, dict[str, float]
                ]
            ],
            baseline_curve: Baseline, name: str = None) -> None:
        if not isinstance(baseline_curve, list):
            baseline_curve = [baseline_curve]

        self.baseline_curve = baseline_curve
        self.metrics_by_debiasing_technique = metrics_by_debiasing_technique
        self.set_name(name)

    def get_debias_methodologies(self) -> list[str]:
        """Returns: list[str]: The list of debiasing methodologies for 
        which this class has recorded metrics
        """
        return list(self.metrics_by_debiasing_technique.keys())

    def get_error_metric_names(self) -> list[str]:
        """Returns: list[str]: The list of error metrics this class has 
        recorded for the debiasing methodologies
        Useful for identifying the correct keywords to specify a graph
        """
        some_debias_method = list(
            self.metrics_by_debiasing_technique.keys())[0]
        return list(self.metrics_by_debiasing_technique[some_debias_method][
            "error"].keys())

    def get_bias_metric_names(self) -> list[str]:
        """Returns: The list of bias metrics this class has recorded for 
        the debiasing methodologies
        Useful for identifying the correct keywords to specify a graph
        """
        some_debias_method = list(
            self.metrics_by_debiasing_technique.keys())[0]
        return list(
            self.metrics_by_debiasing_technique[some_debias_method][
                "fairness"].keys())

    def get_raw_data(self) -> dict[str, dict[str, dict[str, float]]]:
        """Returns: The full dictionary of recorded metrics by debiasing 
        techniques. This has format: debiasing technique name 
        -> ['error', 'fairness'] -> metric name -> metric values
        """
        return self.metrics_by_debiasing_technique

    def set_name(self, name: str) -> None:
        """Updates the recorded name, important for combining graphs so 
        you can identify the origin of the plots
        """
        self.name = name
        if len(self.baseline_curve) == 1:
            self.baseline_curve[0].name = name

    def get_single_graph(
            self, error_metric: str, fairness_metric: str, 
            showlegend: bool = True, include_fairea_labels: bool = True,
            error_bars_type: str = "confidence interval") -> go.Figure:
        """Returns the plotly figure of a graph showing the impact of 
        debiasing as measured using the specified metrics

        Args:
            error_metric (str): which error metric to use on the plot, 
            call get_error_metric_names to find available values
            fairness_metric (str): which bias metric to use on the plot, 
            call get_bias_metric_names to find available values

        Returns:
            go.Figure: The requested plot as a plotly figure
        """
        data = self.__create_scatter_with_baseline(
            error_metric, 
            fairness_metric, 
            showlegend=showlegend, 
            include_fairea_labels=include_fairea_labels,
            error_bars_type=error_bars_type
        )

        layout = go.Layout(
            xaxis_title=f"Bias ({fairness_metric})",
            yaxis_title=f"Error ({error_metric})"
        )

        figure = go.Figure(data=data, layout=layout)
        figure.update_layout(
            title=(f"Debias Methodologies impact on {error_metric} and" 
                   f"{fairness_metric}"))
        figure.update_xaxes(tick0=0, dtick=0.1)
        figure.update_yaxes(tick0=0, dtick=0.1)
        return figure

    def show_single_graph(
            self, error_metric: str, fairness_metric: str, 
            showlegend: bool = True, include_fairea_labels: bool = True,
            error_bars_type: str = "confidence interval") -> None:
        """Displays a graph showing the impact of debiasing as measured 
        using the specified metrics

        Args:
            error_metric (str): which error metric to use on the plot, 
            call get_error_metric_names to find available values
            fairness_metric (str): which bias metric to use on the plot, 
            call get_bias_metric_names to find available values
        """
        self.get_single_graph(error_metric, fairness_metric,
                              showlegend, include_fairea_labels, 
                              error_bars_type).show()

    def get_subplots(
            self, error_metrics: list[str], fairness_metrics: list[str], 
            showlegend: bool = True,
            include_fairea_labels: bool = True, 
            error_bars_type: str = "confidence interval") -> go.Figure:
        """Returns the plotly figure of a matrix of graphs showing the 
        impact of debiasing as measured using the specified metrics

        Args:
            error_metrics (list[str]): which error metrics to use on the 
            plot, call get_error_metric_names to find available values
            fairness_metrics (list[str]): which bias metrics to use on 
            the plot, call get_bias_metric_names to find available 
            values

        Returns:
            go.Figure: The requested plot as a plotly figure
        """
        if isinstance(error_metrics, str):
            error_metrics = [error_metrics]
        if isinstance(fairness_metrics, str):
            fairness_metrics = [fairness_metrics]

        subplots = make_subplots(len(error_metrics), 
                                 len(fairness_metrics))

        for row, error_metric in enumerate(error_metrics):
            for col, fairness_metric in enumerate(fairness_metrics):
                plots = self.__create_scatter_with_baseline(
                    error_metric, 
                    fairness_metric, 
                    showlegend=(row == 0 and col == 0 and showlegend),
                    include_fairea_labels=include_fairea_labels, 
                    error_bars_type=error_bars_type
                )

                for plot in plots:
                    # subplots is 1 indexed while enumerate 0 is indexed
                    subplots.add_trace(plot, row=row+1, col=col+1)

                # Standardise the spacing of ticks across plots to make 
                # easier to interpret at a glance
                subplots.update_xaxes(tick0=0, dtick=0.1, row=row+1, col=col+1)
                subplots.update_yaxes(tick0=0, dtick=0.1, row=row+1, col=col+1)

                if col == 0:
                    subplots.update_yaxes(
                        title_text=f"Performance ({error_metric})", col=col+1, 
                        row=row+1)
                if row == len(error_metrics) - 1:
                    subplots.update_xaxes(
                        title_text=f"Bias ({fairness_metric})", col=col+1, 
                        row=row+1)

        subplots.update_layout(height=400*len(error_metrics),
                               width=600*len(fairness_metrics))
        return subplots

    def show_subplots(
            self, error_metrics: list[str], fairness_metrics: list[str], 
            showlegend: bool = True, include_fairea_labels: bool = True, 
            error_bars_type: str = "confidence interval") -> None:
        """Displays a matrix of graphs showing the impact of debiasing 
        as measured using the specified metrics

        Args:
            error_metrics (list[str]): which error metrics to use on the 
            plot, call get_error_metric_names to find available values
            fairness_metrics (list[str]): which bias metrics to use on 
            the plot, call get_bias_metric_names to find available 
            values
        """
        self.get_subplots(error_metrics, fairness_metrics,
                          showlegend, include_fairea_labels, 
                          error_bars_type=error_bars_type).show()

    def get_all_subplots(
            self, showlegend: bool = True, include_fairea_labels: bool = True, 
            error_bars_type: str = "confidence interval") -> go.Figure:
        """Returns the plotly figure of a matrix of graphs showing the 
        impact of debiasing as measured using all available metrics
        """
        return self.get_subplots(self.get_error_metric_names(), 
                                 self.get_bias_metric_names(), 
                                 showlegend, include_fairea_labels, 
                                 error_bars_type)

    def show_all_subplots(
            self, showlegend: bool = True, include_fairea_labels: bool = True, 
            error_bars_type: str = "confidence interval") -> None:
        """Displays a matrix of graphs showing the impact of debiasing 
        as measured using all available metrics
        """
        self.get_all_subplots(
            showlegend, include_fairea_labels, error_bars_type).show()

    def __create_debias_methodology_point(
            self, debias_methodology: str, error_metric: str, 
            fairness_metric: str, color: str, showlegend: bool = True, 
            value_type: str = "value", 
            error_bars_type: str = "confidence interval") -> go.Scatter:
        """Returns a plotly scatterplot object for a single point: the 
        results of debias_methodology as measured using error_metric and 
        fairness_metric to calculate value

        Args:
            debias_methodology (str): Which of the available debias 
            methodology this point represents
            error_metric (str): Which metric is used to measure error
            fairness_metric (str): Which metric is used to measure bias
            color (str): Which color should be used to represent this 
            point on a graph derived from the returned object
            showlegend (bool, optional): If this point should be 
            included on a legend in a resulting graph. Useful for 
            subplots so that you don't end up with a repeating legend. 
            Defaults to True.
            value_type (str, optional): The type of value to plot for 
            the central point. 
            Available values are "value", "mean", and "median"
            error_bars_type (str, optional): Which of the possible error 
            measurements should be used. 
            Available values are "confidence interval", "range", and 
            "inter quartile range"

        Returns:
            go.Scatter: The point to be placed on a graph
        """

        # Statistics regarding the debiasing technique to be plotted as 
        # a single point
        debias_x = self.metrics_by_debiasing_technique[
            debias_methodology]["fairness"][fairness_metric][value_type]
        debias_y = self.metrics_by_debiasing_technique[
            debias_methodology]["error"][error_metric][value_type]

        # Error bars to put on the single point
        error_bars_x_minus, error_bars_x_plus = (
            self.metrics_by_debiasing_technique[debias_methodology][
                "fairness"][fairness_metric][error_bars_type]
        )
        error_bars_x_minus = debias_x - error_bars_x_minus
        error_bars_x_plus = error_bars_x_plus - debias_x

        error_bars_y_minus, error_bars_y_plus = (
            self.metrics_by_debiasing_technique[
            debias_methodology]["error"][error_metric][error_bars_type]
        )
        error_bars_y_minus = debias_y - error_bars_y_minus
        error_bars_y_plus = error_bars_y_plus - debias_y

        debias_result = go.Scatter(
            x=[debias_x],
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [error_bars_x_plus],
                "arrayminus": [error_bars_x_minus]
            },
            y=[debias_y],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [error_bars_y_plus],
                "arrayminus": [error_bars_y_minus]
            },
            mode="markers",
            name=f"{debias_methodology} outcome",
            showlegend=showlegend,
            marker_color=color
        )

        return debias_result

    def __create_scatter_with_baseline(
            self, error_metric: str, fairness_metric: str, 
            debias_methodologies: list[str] = None, showlegend=True, 
            include_fairea_labels=True, 
            error_bars_type: str = "confidence interval") -> list[go.Scatter]:
        """Create a scatterplot using the specified arguments, to be 
        plotted on either an individual plot or with subplots

        Args:
            error_metric (str): Which metric is used to measure error
            fairness_metric (str): Which metric is used to measure bias
            debias_methodologies (list[str], optional): Which debias 
            methodologies should be included on the resulting plot. 
            Defaults to all available in the dictionary except 
            "no debiasing".
            showlegend (bool, optional): If this plot should be included 
            on a legend in a resulting graph. Useful for subplots so 
            that you don't end up with a repeating legend. 
            Defaults to True.

        Returns:
            list[go.Scatter]: The list of components to be added to a 
            plotly figure or subplot in order to produce the described 
            graph.
        """

        # Need a large number of colors because we want to be able to 
        # plot the composition of graphs
        colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

        if debias_methodologies is None:
            debias_methodologies = self.get_debias_methodologies()
            # TODO do I want to skip 'no debiasing'? Loses error 
            # measurement and makes combining graphs more difficult
            # If the debias_methodologies to be plotted are not 
            # specified, plot all but 'no debiasing'
            # debias_methodologies.remove("no debiasing")
        elif isinstance(debias_methodologies, str):
            debias_methodologies = [debias_methodologies]

        # TODO quick solution to make the graphs consistent, should be 
        # replaced with full implementation of 
        # DebiasingGraphsComposition. I have attempted to write this 
        # class to fulfil both purposes. for simplicity, however that 
        # was a mistake as it cost maintainability.
        
        # len(debias_methodologies) // len(self.baseline_curve) is the 
        # number of debiasing methodologies per component graph, add 1 
        # for the curve itself
        curve_color_indexes = [i * (len(debias_methodologies) // len(
            self.baseline_curve) + 1) for i in range(len(self.baseline_curve))]
        baseline_curves_figures = []

        for i in range(len(self.baseline_curve)):
            baseline_curves_figures.append(
                self.baseline_curve[i].get_baseline_curve(
                    error_metric, fairness_metric, 
                    colors[curve_color_indexes[i]], showlegend, 
                    include_fairea_labels))

        curve_color_indexes = set(curve_color_indexes)
        colors = [color for index, color in enumerate(
            colors) if index not in curve_color_indexes]

        scatter_plots = [
            self.__create_debias_methodology_point(
                method, error_metric, fairness_metric, color=color, 
                showlegend=showlegend, error_bars_type=error_bars_type
            ) for method, color in zip(debias_methodologies, colors)
        ]
        
        return [*baseline_curves_figures, *scatter_plots]

    def __add__(self, other) -> "DebiasingGraphsComposition":
        if isinstance(other, DebiasingGraphsObject):
            return DebiasingGraphsComposition(self, other)
        return NotImplemented


class DebiasingGraphsComposition:
    """Wrapper around DebiasingGraphsObject to make it easier to combine 
    plots without losing information 
    """

    def __init__(self, *debiasing_graphs: DebiasingGraphsObject) -> None:
        self.debiasing_graphs = debiasing_graphs
        self.composite_debias_graph = None

    def __add__(self, other):
        if isinstance(other, DebiasingGraphsComposition):
            return DebiasingGraphsComposition(*self.debiasing_graphs, 
                                              *other.debiasing_graphs)
        if isinstance(other, DebiasingGraphsObject):
            return DebiasingGraphsComposition(*self.debiasing_graphs, other)
        return NotImplemented

    def __compute_composite_debiasing_graph(self) -> None:
        metrics_by_debiasing_technique = dict()
        baselines = []

        for graph_object in self.debiasing_graphs:
            name = graph_object.name
            data = graph_object.get_raw_data()

            # Combine the dictionaries of recorded metrics, updating the 
            # keys to ensure that names do not overlap
            for key, value in data.items():
                if name:
                    key = name + "<br>" + key

                i = 1
                updated_key = key
                while updated_key in metrics_by_debiasing_technique.keys():
                    updated_key = key + " " + str(i)
                    i += 1

                metrics_by_debiasing_technique[updated_key] = value

            baselines.extend(graph_object.baseline_curve)

        self.composite_debias_graph = DebiasingGraphsObject(
            metrics_by_debiasing_technique, baselines)

    def __getattr__(self, attr):
        # Handles method calls intended for self.composite_debias_graph
        if self.composite_debias_graph is None:
            self.__compute_composite_debiasing_graph()
        if hasattr(self.composite_debias_graph, attr):
            return getattr(self.composite_debias_graph, attr)

        raise AttributeError(f"No attribute '{attr}'")
