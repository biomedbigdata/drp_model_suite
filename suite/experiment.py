from typing import Dict, List, Optional, Tuple
import warnings

from .dataset import DrugResponseDataset, FeatureDataset
import pandas as pd
from .evaluation import evaluate
from .drp_model import DRPModel
from ray import tune
import numpy as np
import os


def drug_response_experiment(
    models: List[DRPModel],
    response_data: DrugResponseDataset,
    multiprocessing: bool = False,
    test_mode: str = "LPO",
    randomization_test_views: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[DrugResponseDataset]]:
    """
    Run the drug response prediction experiment.
    :param models: list of models to compare
    :param response_data: drug response dataset
    :param multiprocessing: whether to use multiprocessing
    :param randomization_test_views: views to use for the randomization tests. Key is the name of the randomization test and the value is a list of views to randomize
            e.g. {"randomize_genomics": ["copy_number_var", "mutation"], "methylation_only": ["gene_expression", "copy_number_var", "mutation"]}"
    :return: dictionary containing the results
    """
    results = {model.model_name: {} for model in models}

    for model in models:
        results[model.model_name] = {"predictions": []}
        if randomization_test_views:
            results[model.model_name]["randomization_tests"] = []

        model_hpam_set = model.get_hyperparameter_set()

        response_data.split_dataset(
            n_cv_splits=5,
            mode=test_mode,
            split_validation=True,
            validation_ratio=0.1,
            random_state=42,
        )

        for split in response_data.cv_splits:
            train_dataset = split["train"]
            validation_dataset = split["validation"]
            test_dataset = split["test"]

            # if model.early_stopping is true then we split the validation set into a validation and early stopping set
            if model.early_stopping:
                validation_dataset, early_stopping_dataset = split_early_stopping(
                    validation_dataset=validation_dataset, test_mode=test_mode
                )

            if multiprocessing:
                best_hpams = hpam_tune_raytune(
                    model=model,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    early_stopping_dataset=(
                        early_stopping_dataset if model.early_stopping else None
                    ),
                    hpam_set=model_hpam_set,
                )
            else:
                best_hpams = hpam_tune(
                    model=model,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    early_stopping_dataset=(
                        early_stopping_dataset if model.early_stopping else None
                    ),
                    hpam_set=model_hpam_set,
                )
            train_dataset.add_rows(
                validation_dataset
            )  # use full train val set data for final training
            train_dataset.shuffle(random_state=42)

            test_dataset = train_and_predict(
                model=model,
                hpams=best_hpams,
                train_dataset=train_dataset,
                prediction_dataset=test_dataset,
                early_stopping_dataset=(
                    early_stopping_dataset if model.early_stopping else None
                ),
            )
            results[model.model_name]["predictions"].append(test_dataset)

            if randomization_test_views:
                r = randomization_test(
                    randomization_test_views=randomization_test_views,
                    model=model,
                    hpam_set=best_hpams,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    early_stopping_dataset=early_stopping_dataset,
                )
                results[model.model_name]["randomization_tests"].append(r)

    return results


def randomization_test(
    randomization_test_views: Dict[str, List[str]],
    model: DRPModel,
    hpam_set: Dict,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
) -> Dict:
    cl_features = model.get_cell_line_features(path=hpam_set["feature_path"])
    drug_features = model.get_drug_features(path=hpam_set["feature_path"])
    results = {}
    for test_name, views in randomization_test_views.items():
        for view in views:
            cl_features_rand = cl_features.copy()
            drug_features_rand = drug_features.copy()
            if view in cl_features.get_view_names():
                cl_features.randomize_features(view, mode="gaussian")
            elif view in drug_features.get_view_names():
                drug_features.randomize_features(view, mode="gaussian")
            else:
                warnings.warn(
                    f"View {view} not found in features. Skipping randomization test {test_name} which includes this view."
                )
                break
            test_dataset_rand = train_and_predict(
                model=model,
                hpams=hpam_set,
                train_dataset=train_dataset,
                prediction_dataset=test_dataset,
                early_stopping_dataset=early_stopping_dataset,
                cl_features=cl_features_rand,
                drug_features=drug_features_rand,
            )
            results[test_name] = test_dataset_rand
    return results


def split_early_stopping(
    validation_dataset: DrugResponseDataset, test_mode: str
) -> Tuple[DrugResponseDataset]:
    validation_dataset.shuffle(random_state=42)
    cv_v = validation_dataset.split_dataset(
        n_cv_splits=4,
        mode=test_mode,
        split_validation=False,
        random_state=42,
    )
    # take the first fold of a 4 cv as the split ie. 3/4 for validation and 1/4 for early stopping
    validation_dataset = cv_v[0]["train"]
    early_stopping_dataset = cv_v[0]["test"]
    return validation_dataset, early_stopping_dataset


def train_and_predict(
    model: DRPModel,
    hpams: Dict[str, List],
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    cl_features: Optional[FeatureDataset] = None,
    drug_features: Optional[FeatureDataset] = None,
) -> DrugResponseDataset:
    if cl_features is None:
        cl_features = model.get_cell_line_features(path=hpams["feature_path"])
    if drug_features is None:
        drug_features = model.get_drug_features(path=hpams["feature_path"])
    # making sure there are no missing features:
    train_dataset.reduce_to(
        cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
    )

    prediction_dataset.reduce_to(
        cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
    )

    if early_stopping_dataset is not None:
        early_stopping_dataset.reduce_to(
            cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
        )

    model.train(
        cell_line_input=cl_features,
        drug_input=drug_features,
        output=train_dataset,
        hyperparameters=hpams,
        output_earlystopping=early_stopping_dataset,
    )

    prediction_dataset.predictions = model.predict(
        cell_line_ids=prediction_dataset.cell_line_ids,
        drug_ids=prediction_dataset.drug_ids,
        cell_line_input=cl_features,
        drug_input=drug_features,
    )

    return prediction_dataset


def train_and_evaluate(
    model: DRPModel,
    hpams: Dict[str, List],
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    metric: str = "rmse",
) -> float:
    validation_dataset = train_and_predict(
        model=model,
        hpams=hpams,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
    )
    return evaluate(validation_dataset, metric=[metric])


def hpam_tune(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    hpam_set: List[Dict],
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
) -> Dict:
    best_rmse = float("inf")
    best_hyperparameters = None
    for hyperparameter in hpam_set:
        rmse = train_and_evaluate(
            model=model,
            hpams=hyperparameter,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric="rmse",
        )["rmse"]
        if rmse < best_rmse:
            print(f"current best rmse: {np.round(rmse, 3)}")
            best_rmse = rmse
            best_hyperparameters = hyperparameter
    return best_hyperparameters


def hpam_tune_raytune(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    hpam_set: List[Dict],
) -> Dict:
    analysis = tune.run(
        lambda hpams: train_and_evaluate(
            model=model,
            hpams=hpams,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric="rmse",
        ),
        config=tune.grid_search(hpam_set),
        mode="min",
        num_samples=len(hpam_set),
        resources_per_trial={"cpu": 1},
        chdir_to_trial_dir=False,
        verbose=0,
    )
    best_config = analysis.get_best_config(metric="rmse", mode="min")
    return best_config
