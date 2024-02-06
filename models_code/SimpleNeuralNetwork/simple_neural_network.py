from typing import Optional
from models_code.SimpleNeuralNetwork.utils import FeedForwardNetwork
from suite.model_wrapper import DRPModel
from suite.data_wrapper import DrugResponseDataset, FeatureDataset
import numpy as np
import pandas as pd


class SimpleNeuralNetwork(DRPModel):
    """
    Simple Feedforward Neural Network model with dropout.
    hyperparameters:
        units_per_layer: number of units per layer e.g. [100, 50] means 2 layers with 100 and 50 units respectively and the output layer with one unit.
        dropout_prob: dropout probability for layers 1, 2, ..., n-1
    """

    cell_line_views = ["gene_expression"]

    drug_views = ["fingerprints"]

    def build_model(self, *args, **kwargs):
        """
        Builds the model.
        """
        pass

    def get_feature_matrix(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
    ):
        X_drug = drug_input.get_feature_matrix("fingerprints", drug_ids)
        X_cell_line = cell_line_input.get_feature_matrix(
            "gene_expression", cell_line_ids
        )
        return np.concatenate((X_drug, X_cell_line), axis=1)

    def train(
        self,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
        output: DrugResponseDataset,
        hyperparameters: dict,
        cell_line_input_earlystopping: Optional[FeatureDataset] = None,
        drug_input_earlystopping: Optional[FeatureDataset] = None,
        output_earlystopping: Optional[DrugResponseDataset] = None,
    ):
        """
        Trains the model.
        :param cell_line_input: training data associated with the cell line input
        :param drug_input: training data associated with the drug input
        :param output: training data associated with the reponse output
        """
        X = self.get_feature_matrix(
            cell_line_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        if (
            cell_line_input_earlystopping
            and drug_input_earlystopping
            and output_earlystopping
        ):
            X_earlystopping = self.get_feature_matrix(
                drug_input_earlystopping,
                cell_line_input_earlystopping,
                output_earlystopping,
            )
        else:
            X_earlystopping = None

        neural_network = FeedForwardNetwork(
            n_features=X.shape[1],
            n_units_per_layer=hyperparameters["units_per_layer"],
            dropout_prob=hyperparameters["dropout_prob"],
        )
        if output_earlystopping:
            response_earlystopping = output_earlystopping.response
        else:
            response_earlystopping = None
        neural_network.fit(
            X,
            output.response,
            X_earlystopping,
            response_earlystopping,
            batch_size=64,
            patience=5,
            num_workers=2,
        )
        self.model = neural_network

    def save(self, path: str):
        """
        Saves the model.
        :param path: path to save the model
        """
        self.model.save(path)

    @staticmethod
    def load(path: str):
        # TODO
        raise NotImplementedError("load method not implemented")

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
    ) -> np.ndarray:
        """
        Predicts the response for the given input. Call the respective function from models_code here.
        :param cell_line_input: input associated with the cell line
        :param drug_input: input associated with the drug
        :return: predicted response
        """
        X = self.get_feature_matrix(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        return self.model.predict(X)

    def get_cell_line_features(self, path: str) -> FeatureDataset:
        """
        Fetch cell line input data
        :return: FeatureDataset
        """
        ge = pd.read_csv(f"{path}/gene_expression.csv", index_col=0)
        landmark_genes = pd.read_csv(f"{path}/gene_lists/landmark_genes.csv", sep="\t")
        genes_to_use = set(landmark_genes["Symbol"]) & set(ge.columns)
        ge = ge[list(genes_to_use)]

        return FeatureDataset(
            {cl: {"gene_expression": ge.loc[cl].values} for cl in ge.index}
        )

    def get_drug_features(self, path: str) -> FeatureDataset:
        """
        Fetch drug input data.
        :return: FeatureDataset
        """
        fingerprints = pd.read_csv(
            f"{path}/drug_fingerprints/drug_name_to_demorgan_128_map.csv", index_col=0
        ).T
        return FeatureDataset(
            {
                drug: {"fingerprints": fingerprints.loc[drug].values}
                for drug in fingerprints.index
            }
        )

    def get_hyperparameter_set(self):
        hpams = [
            {"dropout_prob": 0.2, "units_per_layer": [10, 10, 10]},
            {"dropout_prob": 0.3, "units_per_layer": [20, 10, 10]},
        ]
        for hpam in hpams:
            hpam["feature_path"] = "data/GDSC"
        return hpams
