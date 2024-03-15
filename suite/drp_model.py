from abc import ABC, abstractmethod
from typing import List, Optional
from .dataset import DrugResponseDataset, FeatureDataset
import numpy as np


class DRPModel(ABC):
    """
    Abstract wrapper class for drug response prediction models.
    """
    early_stopping = False

    def __init__(self, model_name, target, *args, **kwargs):
        """
        Creates an instance of a drug response prediction model.
        :param model_name: model name for displaying results
        :param target: target value, e.g., IC50, EC50, AUC, classification
        :param args: optional arguments
        :param kwargs: optional keyword arguments
        """
        self.model_name = model_name
        self.target = target
        self.build_model(*args, **kwargs)

    @abstractmethod
    def get_hyperparameter_set(self) -> List[dict]:
        """
        :return: hyperparameter set list of dicts
        """
        pass

    @property
    @abstractmethod
    def cell_line_views(self):
        """
        Returns the sources the model needs as input for describing the cell line.
        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression", "mutation"]
        """
        pass

    @property
    @abstractmethod
    def drug_views(self):
        """
        Returns the sources the model needs as input for describing the drug.
        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """
        pass

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """
        Builds the model.
        """
        pass

    @abstractmethod
    def train(
        self,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
        output: DrugResponseDataset,
        output_earlystopping: Optional[DrugResponseDataset] = None,
    ):
        """
        Trains the model. Call the respective function from models_code here.
        :param cell_line_input: training data associated with the cell line input
        :param drug_input: training data associated with the drug input
        :param output: training data associated with the response output
        :param output_earlystopping: optional early stopping dataset
        """
        pass

    @abstractmethod
    def predict(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset):
        """
        Predicts the response for the given input. Call the respective function from models_code here.
        :param cell_line_input: input associated with the cell line
        :param drug_input: input associated with the drug
        :return: predicted response
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Saves the model to models_saved.
        :param path: path to save the model
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Loads the model.
        :param path: path to load the model
        """
        pass

    @abstractmethod
    def get_cell_line_features(self, cell_line_input: FeatureDataset):
        """
        :return: FeatureDataset
        """
        pass

    @abstractmethod
    def get_drug_features(self, drug_input: FeatureDataset):
        """
        :return: FeatureDataset
        """
        pass


class SingleDRPModel(DRPModel, ABC):
    """
    Abstract wrapper class for single drug response prediction models.
    """

    def __init__(self, model_name, target, config_path, *args, **kwargs):
        """
        Creates an instance of a single drug response prediction model.
        :param model_name: model name for displaying results
        :param target: target value, e.g., IC50, EC50, AUC, classification
        :param args: optional arguments
        :param kwargs: optional keyword arguments
        """
        super(SingleDRPModel, self).__init__(model_name, target, config_path, *args, **kwargs)

    @property
    @abstractmethod
    def cell_line_views(self):
        """
        Returns the sources the model needs as input for describing the cell line.
        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression", "mutation"]
        """
        pass

    @property
    @abstractmethod
    def drug_views(self):
        """
        Returns the sources the model needs as input for describing the drug.
        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """
        pass

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """
        Builds the model.
        """
        pass

    def train(
        self,
        cell_line_input: FeatureDataset,
        drug_input: str,
        output: DrugResponseDataset,
    ):
        """
        Trains the model.
        :param cell_line_input: training data associated with the cell line input
        :param drug_input: drug name
        :param output: training data associated with the response output
        """
        self.train_drug(cell_line_input, drug_input, output)

    @abstractmethod
    def train_drug(
        self,
        cell_line_input: FeatureDataset,
        drug_name: str,
        output: DrugResponseDataset,
    ):
        """
        Trains one model per drug.
        :param cell_line_input: training data associated with the cell line input
        :param drug_name: drug name
        :param output: training data associated with the response output
        """
        pass

    def predict(self, cell_line_input: FeatureDataset, drug_input: str) -> np.ndarray:
        """
        Predicts the response for the given input.
        :param cell_line_input: input associated with the cell line
        :param drug_input: drug name
        :return: predicted response
        """
        self.predict_drug(cell_line_input, drug_input)

    @abstractmethod
    def predict_drug(self, cell_line_input: FeatureDataset, drug_name: str):
        """
        Predicts the response for the given single drug.
        :param cell_line_input: input associated with the cell line
        :param drug_name: drug name
        :return: predicted response
        """
        raise NotImplementedError("predict_drug method not implemented")

    @abstractmethod
    def save(self, path):
        """
        Saves the model to models_saved.
        :param path: path to save the model
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Loads the model.
        :param path: path to load the model
        """
        pass
