from .dataset import DrugResponseDataset
from typing import Union, List
import numpy as np
import sklearn.metrics as metrics
from .utils import pearson, spearman, kendall, partial_correlation

AVAILABLE_METRICS = {
    "mse": metrics.mean_squared_error,
    "rmse": metrics.mean_squared_error,
    "mae": metrics.mean_absolute_error,
    "r2": metrics.r2_score,
    "pearson": pearson,
    "spearman": spearman,
    "kendall": kendall,
    "partial_correlation": partial_correlation,
}


def evaluate(dataset: DrugResponseDataset, metric: Union[List[str], str]):
    """
    Evaluates the model on the given dataset.
    :param dataset: dataset to evaluate on
    :param metric: evaluation metric(s) (one or a list of "mse", "rmse", "mae", "r2", "pearson", "spearman", "kendall", "partial_correlation")
    :return: evaluation metric
    """
    if isinstance(metric, str):
        metric = [metric]
    predictions = dataset.predictions
    response = dataset.response
    results = {}
    for m in metric:
        assert (
            m in AVAILABLE_METRICS
        ), f"invalid metric {m}. Available: {list(AVAILABLE_METRICS.keys())}"
        if m == "partial_correlation":
            results[m] = AVAILABLE_METRICS[m](
                predictions, response, dataset.cell_line_ids, dataset.drug_ids
            )
        else:
            results[m] = AVAILABLE_METRICS[m](predictions, response)

    return results


def visualize_results(dataset: DrugResponseDataset, mode: Union[List[str], str]):
    """
    Visualizes the model on the given dataset.
    :param dataset: dataset to evaluate on
    :mode
    :return: evaluation metric
    """
    raise NotImplementedError("visualize not implemented yet")
