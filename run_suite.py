# from models import SimpleNeuralNetwork
# import pandas as pd
# neural_net_baseline = SimpleNeuralNetwork("smpl", target="IC50")

# models = [neural_net_baseline]

# response_data = pd.read_csv("data/GDSC/response_GDSC2.csv")
# output = response_data["LN_IC50"].values
# cell_line_ids = response_data["CELL_LINE_NAME"].values
# drug_ids = response_data["DRUG_NAME"].values
# response_data = DrugResponseDataset(
#     response=output, cell_line_ids=cell_line_ids, drug_ids=drug_ids
# )
# result = drug_response_experiment(models, response_data, multiprocessing=True, randomization_test_views={"randomize_gene_expression": ["gene_expression"]})
# print(result)

import argparse
import pandas as pd
import numpy as np
import json

from models import MODEL_FACTORY
from suite.dataset import DrugResponseDataset
from suite.experiment import drug_response_experiment


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run the drug response prediction model test suite"
    )
    parser.add_argument(
        "--run_id", type=str, default="", help="identifier to save the results"
    )
    parser.add_argument(
        "--models", nargs="+", help="model to evalaute or list of models to compare"
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default="LPO",
        help="Which tests to run (LPO=Leave-random-Pairs-Out, "
        "LCO=Leave-Cell-line-Out, LDO=Leave-Drug-Out)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="GDSC1",
        help="Name of the drug response dataset"
    )

    parser.add_argument(
        "--path_out", type=str, default="results/", help="Path to the output directory"
    )
    parser.add_argument(
        "--curve_curator",
        action='store_true',
        default=False,
        help="Whether to run " "CurveCurator " "to sort out " "non-reactive " "curves",
    )
    parser.add_argument(
        "--optim_metric",
        type=str,
        default="RMSE",
        help="Metric to optimize for (RMSE, AUC, ACC, F1, MCC, R2, etc.)"
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    assert args.models, "At least one model must be specified"
    assert all(
        [model in MODEL_FACTORY for model in args.models]
    ), f"Invalid model name. Available models are {list(MODEL_FACTORY.keys())}"
    assert args.test_mode in [
        "LPO",
        "LCO",
        "LDO",
    ], f"Invalid test mode. Available test modes are 'LPO', 'LCO', 'LDO'"
    models = [
        MODEL_FACTORY[model](model_name=model, target="IC50") for model in args.models
    ]

    # TODO like the models we want to have a DATASET_FACTORY which loads and optionally preprocesses the dataset
    if args.dataset_name == "GDSC1":
        response_data = pd.read_csv("data/GDSC/response_GDSC1.csv")
        output = response_data["LN_IC50"].values
        cell_line_ids = response_data["CELL_LINE_NAME"].values
        drug_ids = response_data["DRUG_NAME"].values
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} not implemented")

    if args.curve_curator:
        raise NotImplementedError("CurveCurator not implemented")

    response_data = DrugResponseDataset(
        response=output, cell_line_ids=cell_line_ids, drug_ids=drug_ids
    )
    # TODO randomization_test_views need to be specified. maybe via config file 
    result = drug_response_experiment(
        models,
        response_data,
        multiprocessing=True,
        test_mode=args.test_mode,
        randomization_test_views={"randomize_gene_expression": ["gene_expression"], "randomize_genomics": ["mutation", "copy_number_var"]},
    )

    # TODO now do evaluation, visualization, etc.
    # Convert to JSON string
    if False:
        # DrugResponseDataset not serializable
        json_string = json.dumps(result, indent=4)

        # Save JSON string to a file
        with open(f"{args.path_out}/{args.run_id}_results.npy", "w") as json_file:
            json_file.write(json_string)
        print(f"Done! Results saved to {args.path_out}/{args.run_id}_results.npy")
    print(result)