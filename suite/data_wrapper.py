from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np
from numpy.typing import ArrayLike
from .utils import leave_pair_out_cv, leave_group_out_cv


class Dataset(ABC):
    """
    Abstract wrapper class for datasets.
    """

    @abstractmethod
    def load(self):
        """
        Loads the dataset from data.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Saves the dataset to data.
        """
        pass


class DrugResponseDataset(Dataset):
    """
    Drug response dataset.
    """

    def __init__(
        self,
        response: ArrayLike,
        cell_line_ids: ArrayLike,
        drug_ids: ArrayLike,
        *args,
        **kwargs,
    ):
        """
        Initializes the drug response dataset.
        :param response: drug response values per cell line and drug
        :param cell_line_ids: cell line IDs
        :param drug_ids: drug IDs

        Variables:
        response: drug response values per cell line and drug
        cell_line_ids: cell line IDs
        drug_ids: drug IDs
        predictions: optional. Predicted drug response values per cell line and drug
        """
        super(DrugResponseDataset, self).__init__()
        self.response = np.array(response)
        self.cell_line_ids = np.array(cell_line_ids)
        self.drug_ids = np.array(drug_ids)
        self.predictions = None

    def __len__(self):
        return len(self.response)

    def __str__(self):
        if len(self.response) > 3:
            string = f"DrugResponseDataset: CLs {self.cell_line_ids[:3]}...; Drugs {self.drug_ids[:3]}...; Response {self.response[:3]}..."
        else:
            string = f"DrugResponseDataset: CLs {self.cell_line_ids}; Drugs {self.drug_ids}; Response {self.response}"
        if self.predictions is not None:
            if len(self.predictions) > 3:
                string += f"; Predictions {self.predictions[:3]}..."
            else:
                string += f"; Predictions {self.predictions}"
        return string

    def load(self):
        """
        Loads the drug response dataset from data.
        """
        raise NotImplementedError("load method not implemented")

    def save(self):
        """
        Saves the drug response dataset to data.
        """
        raise NotImplementedError("save method not implemented")

    def add_rows(self, other: "DrugResponseDataset") -> None:
        """
        Adds rows from another dataset.
        :other: other dataset
        """
        self.response = np.concatenate([self.response, other.response])
        self.cell_line_ids = np.concatenate([self.cell_line_ids, other.cell_line_ids])
        self.drug_ids = np.concatenate([self.drug_ids, other.drug_ids])

        if self.predictions is not None and other.predictions is not None:
            self.predictions = np.concatenate([self.predictions, other.predictions])

    def shuffle(self, random_state: int = 42) -> None:
        """
        Shuffles the dataset.
        :random_state: random state
        """
        indices = np.arange(len(self.response))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        self.response = self.response[indices]
        self.cell_line_ids = self.cell_line_ids[indices]
        self.drug_ids = self.drug_ids[indices]
        if self.predictions is not None:
            self.predictions = self.predictions[indices]

    def remove_drugs(self, drugs_to_remove: Union[str, list]) -> None:
        """
        Removes drugs from the dataset.
        :drugs_to_remove: name of drug or list of names of multiple drugs to remove
        """
        if isinstance(drugs_to_remove, str):
            drugs_to_remove = [drugs_to_remove]

        mask = [drug not in drugs_to_remove for drug in self.drug_ids]
        self.drug_ids = self.drug_ids[mask]
        self.cell_line_ids = self.cell_line_ids[mask]
        self.response = self.response[mask]

    def remove_cell_lines(self, cell_lines_to_remove: Union[str, list]) -> None:
        """
        Removes cell lines from the dataset.
        :cell_lines_to_remove: name of cell line or list of names of multiple cell lines to remove
        """
        if isinstance(cell_lines_to_remove, str):
            cell_lines_to_remove = [cell_lines_to_remove]

        mask = [
            cell_line not in cell_lines_to_remove for cell_line in self.cell_line_ids
        ]
        self.drug_ids = self.drug_ids[mask]
        self.cell_line_ids = self.cell_line_ids[mask]
        self.response = self.response[mask]

    def reduce_to(self, cell_line_ids: ArrayLike, drug_ids: ArrayLike) -> None:
        """
        Removes all rows which contain a cell_line not in cell_line_ids or a drug not in drug_ids
        :cell_line_ids: cell line IDs
        :drug_ids: drug IDs
        """
        self.remove_drugs([drug for drug in self.drug_ids if drug not in drug_ids])
        self.remove_cell_lines(
            [
                cell_line
                for cell_line in self.cell_line_ids
                if cell_line not in cell_line_ids
            ]
        )

    def split_dataset(
        self,
        n_cv_splits,
        mode,
        split_validation=True,
        validation_ratio=0.1,
        random_state=42,
    ) -> List[dict]:
        """
        Splits the dataset into training, validation and test sets for crossvalidation
        :param mode: split mode (LPO=Leave-random-Pairs-Out, LCO=Leave-Cell-line-Out, LDO=Leave-Drug-Out)
        :return: training, validation and test sets
        """

        cell_line_ids = self.cell_line_ids
        drug_ids = self.drug_ids
        response = self.response

        if mode == "LPO":
            cv_splits = leave_pair_out_cv(
                n_cv_splits,
                response,
                cell_line_ids,
                drug_ids,
                split_validation,
                validation_ratio,
                random_state,
            )

        elif mode in ["LCO", "LDO"]:
            group = "cell_line" if mode == "LCO" else "drug"
            cv_splits = leave_group_out_cv(
                group=group,
                n_cv_splits=n_cv_splits,
                response=response,
                cell_line_ids=cell_line_ids,
                drug_ids=drug_ids,
                split_validation=split_validation,
                validation_ratio=validation_ratio,
                random_state=random_state,
            )
        else:
            raise ValueError(
                f"Unknown split mode '{mode}'. Choose from 'LPO', 'LCO', 'LDO'."
            )
        self.cv_splits = cv_splits  # TODO save these as DrugResponseDatasets !!!
        return cv_splits


class FeatureDataset(Dataset):
    """
    Class for feature datasets.
    """

    def __init__(self, features: Dict[str, Dict[str, np.ndarray]], *args, **kwargs):
        """
        Initializes the feature dataset.
        :features: dictionary of features, key: drug ID, value: Dict of feature views, key: feature name, value: feature vector
        """
        super(FeatureDataset, self).__init__()
        self.features = features
        self.view_names = self.get_view_names()
        self.identifiers = self.get_ids()

    def load(self):
        """
        loads the feature dataset from data.
        """
        raise NotImplementedError("save method not implemented")

    def save(self):
        """
        Saves the feature dataset to data.
        """
        raise NotImplementedError("save method not implemented")

    def randomize_features(
        self, views_to_randomize: Union[str, list], mode: str
    ) -> None:
        """
        Randomizes the feature vectors.
        :views_to_randomize: name of feature view or list of names of multiple feature views to randomize. The other views are not randomized.
        :mode: randomization mode (permutation, gaussian, zeroing)
        """
        if isinstance(views, str):
            views = [views]

        if mode == "permutation":
            # Get the entity names
            identifiers = self.get_ids()

            # Permute the specified views for each entity (= cell line or drug)
            self.features = {
                entity: {
                    view: self.features[entity][view]
                    if view not in views_to_randomize
                    else self.features[other_entity][view]
                    for view, other_entity in zip(
                        self.features[entity].keys(), np.random.permutation(identifiers)
                    )
                }
                for entity in identifiers
            }

        elif mode == "gaussian":
            for view in views:
                for identifier in self.get_ids():
                    self.features[identifier][view] = np.random.normal(
                        self.features[identifier][view].mean(),
                        self.features[identifier][view].std(),
                        self.features[identifier][view].shape,
                    )
        elif mode == "zeroing":
            for view in views:
                for identifier in self.get_ids():
                    self.features[identifier][view] = np.zeros(
                        self.features[identifier][view].shape
                    )
        else:
            raise ValueError(
                f"Unknown randomization mode '{mode}'. Choose from 'permutation', 'gaussian', 'zeroing'."
            )

    def normalize_features(
        self, views: Union[str, list], normalization_parameter
    ) -> None:
        """
        normalize the feature vectors.
        :views: name of feature view or list of names of multiple feature views to normalize. The other views are not normalized.
        :normalization_parameter:
        """
        # TODO
        raise NotImplementedError("normalize_features method not implemented")

    def get_mean_and_standard_deviation(self) -> None:
        """
        get columnwise mean and standard deviation of the feature vectors for all views.
        """
        # TODO
        raise NotImplementedError(
            "get_mean_and_standard_deviation method not implemented"
        )

    def get_ids(self):
        """
        returns drug ids of the dataset
        """
        return list(self.features.keys())

    def get_view_names(self):
        """
        returns feature view names
        """
        return list(self.features[list(self.features.keys())[0]].keys())

    def get_feature_matrix(self, view: str, identifiers: str) -> np.ndarray:
        """
        Returns the feature matrix for the given view.
        :param drug_input: drug input
        :param cell_line_input: cell line input
        :param view: view name
        :return: feature matrix
        """
        assert view in self.view_names, f"View '{view}' not in in the FeatureDataset."
        missing_identifiers = {
            id_ for id_ in identifiers if id_ not in self.identifiers
        }
        assert (
            not missing_identifiers
        ), f"{len(missing_identifiers)} of {len(np.unique(identifiers))} ids are not in the FeatureDataset. Missing ids: {missing_identifiers}"

        return np.stack([self.features[id_][view] for id_ in identifiers], axis=0)
