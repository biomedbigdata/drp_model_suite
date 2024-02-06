from typing import List
from sklearn.model_selection import KFold, GroupKFold
import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split


def leave_pair_out_cv(
    n_cv_splits: int,
    response: ArrayLike,
    cell_line_ids: ArrayLike,
    drug_ids: ArrayLike,
    split_validation=True,
    validation_ratio=0.1,
    random_state=42,
) -> List[dict]:
    """
    Leave pair out cross validation. Splits data into n_cv_splits number of cross validation splits.
    :param n_cv_splits: number of cross validation splits
    :param response: response (e.g. ic50 values)
    :param cell_line_ids: cell line IDs
    :param drug_ids: drug IDs
    :param split_validation: whether to split the training set into training and validation set
    :param validation_ratio: ratio of validation set (of the training set)
    :param random_state: random state
    :return: list of dicts of the cross validation sets
    """
    from .data_wrapper import DrugResponseDataset

    assert (
        len(response) == len(cell_line_ids) == len(drug_ids)
    ), "response, cell_line_ids and drug_ids must have the same length"

    kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=random_state)
    cv_sets = []

    for train_indices, test_indices in kf.split(response):
        if split_validation:
            # split training set into training and validation set
            train_indices, validation_indices = train_test_split(
                train_indices,
                test_size=validation_ratio,
                shuffle=True,
                random_state=random_state,
            )
        cv_fold = {
            "train": DrugResponseDataset(
                cell_line_ids=cell_line_ids[train_indices],
                drug_ids=drug_ids[train_indices],
                response=response[train_indices],
            ),
            "test": DrugResponseDataset(
                cell_line_ids=cell_line_ids[test_indices],
                drug_ids=drug_ids[test_indices],
                response=response[test_indices],
            ),
        }

        if split_validation:
            cv_fold["validation"] = DrugResponseDataset(
                cell_line_ids=cell_line_ids[validation_indices],
                drug_ids=drug_ids[validation_indices],
                response=response[validation_indices],
            )

        cv_sets.append(cv_fold)
    return cv_sets


def leave_group_out_cv(
    group: str,
    n_cv_splits: int,
    response: ArrayLike,
    cell_line_ids: ArrayLike,
    drug_ids: ArrayLike,
    split_validation=True,
    validation_ratio=0.1,
    random_state=42,
):
    from .data_wrapper import DrugResponseDataset

    """
    Leave group out cross validation. Splits data into n_cv_splits number of cross validation splits.
    :param group: group to leave out (cell_line or drug)
    :param n_cv_splits: number of cross validation splits
    :param random_state: random state
    :return: list of dicts of the cross validation sets
    """
    assert group in {
        "cell_line",
        "drug",
    }, f"group must be 'cell_line' or 'drug', but is {group}"

    if group == "cell_line":
        group_ids = cell_line_ids
    elif group == "drug":
        group_ids = drug_ids

    # shuffle, since GroupKFold does not implement this
    indices = np.arange(len(response))
    shuffled_indices = np.random.RandomState(seed=random_state).permutation(indices)
    response = response[shuffled_indices]
    cell_line_ids = cell_line_ids[shuffled_indices]
    drug_ids = drug_ids[shuffled_indices]
    group_ids = group_ids[shuffled_indices]

    gkf = GroupKFold(n_splits=n_cv_splits)
    cv_sets = []

    for train_indices, test_indices in gkf.split(response, groups=group_ids):
        cv_fold = {
            "train": DrugResponseDataset(
                cell_line_ids=cell_line_ids[train_indices],
                drug_ids=drug_ids[train_indices],
                response=response[train_indices],
            ),
            "test": DrugResponseDataset(
                cell_line_ids=cell_line_ids[test_indices],
                drug_ids=drug_ids[test_indices],
                response=response[test_indices],
            ),
        }
        if split_validation:
            # TODO validation set should also not contain the same cell lines or drugs as the training set

            raise NotImplementedError(
                "validation set split not implemented for leave_group_out_cv"
            )
        cv_sets.append(cv_fold)
    return cv_sets


def partial_correlation(
    predictions: np.ndarray,
    response: np.ndarray,
    cell_line_ids: np.ndarray,
    drug_ids: np.ndarray,
) -> float:
    """
    Computes the partial correlation between predictions and response, conditioned on cell line and drug.
    :param predictions: predictions
    :param response: response
    :param cell_line_ids: cell line IDs
    :param drug_ids: drug IDs
    :return: partial correlation float
    """

    assert (
        len(predictions) == len(response) == len(cell_line_ids) == len(drug_ids)
    ), "predictions, response, drug_ids, and cell_line_ids must have the same length"

    raise NotImplementedError("partial correlation not implemented yet")


def pearson(predictions: np.ndarray, response: np.ndarray) -> float:
    """
    Computes the pearson correlation between predictions and response.
    :param predictions: predictions
    :param response: response
    :param cell_line_ids: cell line IDs
    :param drug_ids: drug IDs
    :return: pearson correlation float
    """

    assert len(predictions) == len(
        response
    ), "predictions, response  must have the same length"
    return np.corrcoef(predictions, response)[0, 1]


def spearman(predictions: np.ndarray, response: np.ndarray) -> float:
    """
    Computes the spearman correlation between predictions and response.
    :param predictions: predictions
    :param response: response
    :param cell_line_ids: cell line IDs
    :param drug_ids: drug IDs
    :return: spearman correlation float
    """
    # we can use scipy.stats.spearmanr
    assert len(predictions) == len(
        response
    ), "predictions, response  must have the same length"
    raise NotImplementedError


def kendall(predictions: np.ndarray, response: np.ndarray) -> float:
    """
    Computes the kendall tau correlation between predictions and response.
    :param predictions: predictions
    :param response: response
    :param cell_line_ids: cell line IDs
    :param drug_ids: drug IDs
    :return: kendall tau correlation float
    """
    # we can use scipy.stats.spearmanr
    assert len(predictions) == len(
        response
    ), "predictions, response  must have the same length"
    raise NotImplementedError
