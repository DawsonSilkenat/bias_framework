from .baseline import Baseline
import numpy as np
from ..metrics import get_all_metrics
import plotly.graph_objs as go


class FaireaCurve(Baseline):
    """The fairea curve is a baseline to which debiasing techniques can 
    be compared. By converting a percentage of the models results before 
    debiasing is applied to the most frequent class we get a naive 
    result for the error bias tradeoff. By using a range of percentages 
    we produce a curve to which debiasing techniques can be compared
    """

    def __init__(
            self, true_values: np.array, predicted_values: np.array, 
            privilege_status: np.array, 
            fractions_to_mutate: list[float] = [i/10 for i in range(11)], 
            repetitions: int = 50) -> None:
        super().__init__()
        self.metrics_by_mutation = _fairea_model_mutation(
            true_values, predicted_values, privilege_status, 
            fractions_to_mutate, repetitions)

    def get_baseline_curve(
            self, error_metric: str, fairness_metric: str, color: str, 
            showlegend: bool = True, 
            include_labels: bool = True) -> go.Scatter:
        """Return a plotly object which can be added to a figure to 
        display the fairea curve.

        Args:
            error_metric (str): Which measurement of error is used for 
            the curve
            fairness_metric (str): Which measurement of bias is used for 
            the curve
            color (str): The color to be used in the resulting curve
            showlegend (bool, optional): If the curve should be included 
            in the legend of a graph using this curve. Defaults to True.
            include_labels (bool, optional): Whether the points along 
            the curve should be labelled with F_n, where n is the 
            percentage mutated. Defaults to True.
        Returns:
            go.Scatter: A plotly scatter object representing the 
            specified curve
        """
        fairea_labels = []
        fairea_x = []
        fairea_y = []

        for mutation, metric in self.metrics_by_mutation:
            fairea_labels.append(f"F_{int(mutation * 100)}")
            fairea_x.append(metric["fairness"][fairness_metric])
            fairea_y.append(metric["error"][error_metric])

        return go.Scatter(
            x=fairea_x,
            y=fairea_y,
            mode="lines+markers+text" if include_labels else "lines+markers",
            name=f"{self.name} fairea baseline" if self.name 
                else "fairea baseline",
            text=fairea_labels,
            textposition="bottom right",
            showlegend=showlegend,
            line_color=color
        )


def _fairea_model_mutation(
        true_values: np.array, predicted_values: np.array, 
        privilege_status: np.array, fractions_to_mutate: list[float], 
        repetitions: int, 
        seed: int = None) -> list[tuple[float, dict[str, dict[str, float]]]]:
    # Source: https://solar.cs.ucl.ac.uk/pdf/hort2021fairea.pdf

    rng = np.random.default_rng(seed)

    # The Fairea paper used the mutation strategy where all values were 
    # mutated to the same label, and the label to mutate to was the one 
    # which produced the highest accuracy with 100% mutation. We follow 
    # this suggestion.
    mutate_to_value = np.argmax(np.bincount(true_values))

    # This internal representation of the curve will have format 
    # list[tuple[float, dict[str, dict[str, float]]]]
    # The first float is the mutation fraction, the dictionary are the 
    # metric results for that
    metrics_by_mutation: list[tuple[float, dict[str, dict[str, float]]]] = []

    for mutation_fraction in fractions_to_mutate:
        number_to_mutate = int(mutation_fraction * len(predicted_values))

        # List of metric results from each repetition of the mutation, 
        # to be averaged at the end
        mutation_fraction_metrics = []

        for _ in range(repetitions):
            # Creating the mutated predictions and retrieving metrics
            indexes_to_mutate = rng.choice(
                len(predicted_values), number_to_mutate, replace=False)
            mutated_predictions = np.copy(predicted_values)
            mutated_predictions[indexes_to_mutate] = mutate_to_value

            mutation_fraction_metrics.append(get_all_metrics(
                true_values, mutated_predictions, privilege_status))

        # Average the results across the iterations and append to the results
        mutation_fraction_metrics_averages = dict()
        for metric_type, metric_type_dict in (
                mutation_fraction_metrics[0].items()):
            mutation_fraction_metrics_averages[metric_type] = dict()

            for metric_name in metric_type_dict.keys():
                mutation_fraction_metrics_averages[
                    metric_type][metric_name] = np.mean(
                        [metric[metric_type][metric_name] 
                         for metric in mutation_fraction_metrics]
                    )
                

        metrics_by_mutation.append(
            (mutation_fraction, mutation_fraction_metrics_averages))

    return metrics_by_mutation
