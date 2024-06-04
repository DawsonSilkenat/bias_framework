import plotly.graph_objs as go
from abc import ABC


class Baseline(ABC):
    """This abstract class for baseline curves, primarily defining that 
    a baseline can be given a name (which we would expect to appear in 
    the legend if provided) and has a method which will return a plotly 
    object which can be shown on a graph. Subject to change as I need to 
    flesh out an understanding of alternatives to the fairea curve.
    """

    def __init__(self):
        self.name = None

    def get_baseline_curve(self, error_metric: str, fairness_metric: str, 
                           color: str, showlegend: bool=True) -> go.Scatter:
        raise NotImplementedError
