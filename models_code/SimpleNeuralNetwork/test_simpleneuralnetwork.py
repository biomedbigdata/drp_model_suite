# not a proper test yet, just trying stuff...
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from models_code import SimpleNeuralNetwork
from suite.data_wrapper import DrugResponseDataset
import pandas as pd
from suite.evaluation import evaluate

neural_net_baseline = SimpleNeuralNetwork("smpl", target="IC50")

models = [neural_net_baseline]
nn_hpam_set = [
    {"dropout_prob": 0.2, "units_per_layer": [10, 10, 10]},
    {"dropout_prob": 0.3, "units_per_layer": [20, 10, 10]},
]
hpam_sets = [nn_hpam_set]
feature_path = "../../data/cell_line_input/GDSC/"  # maybe this should be a parameter of the model class, so that the model can load the features itself, but also depends on the response dataset :S
for model, model_hpam_set in zip(models, hpam_sets):
    cl_features = model.get_cell_line_features(path=feature_path)
    drug_features = model.get_drug_features(path=feature_path)

    response_data = pd.read_csv("../../data/response_output/GDSC/response_GDSC2.csv")
    output = response_data["LN_IC50"].values
    cell_line_ids = response_data["CELL_LINE_NAME"].values
    drug_ids = response_data["DRUG_NAME"].values
    response_data = DrugResponseDataset(
        response=output, cell_line_ids=cell_line_ids, drug_ids=drug_ids
    )

    # making sure there are no missing features:
    response_data.reduce_to(
        cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
    )
    response_data.split_dataset(
        n_cv_splits=5,
        mode="LPO",
        split_validation=True,
        validation_ratio=0.1,
        random_state=42,
    )
    print("Starting training...")
    nsplit=0
    for split in response_data.cv_splits:
        print(f"Split: {nsplit}")
        nsplit += 1
        train_dataset = split["train"]
        validation_dataset = split["validation"]

        test_dataset = split["test"]
        for hyperparameter in model_hpam_set:
            print("Training with hyperparameter set: ", hyperparameter)
            model.train(
                cell_line_input=cl_features,
                drug_input=drug_features,
                output=train_dataset,
                hyperparameters=hyperparameter,
            )
            print("Predicting...")
            validation_dataset.predictions = model.predict(
                cell_line_ids=validation_dataset.cell_line_ids,
                drug_ids=validation_dataset.drug_ids,
                cell_line_input=cl_features,
                drug_input=drug_features,
            )

            metric = evaluate(validation_dataset, metric=["rmse"])["rmse"]
            break
        break
        # best_hyperparameter = ....
        # train_dataset.add_rows(validation_dataset)
        # train_dataset.shuffle(random_state=42)
        # model.train(
        #    cell_line_input=cl_features,
        #    drug_input=drug_features,
        #    output=train_dataset,
        #    hyperparameters=best_hyperparameter,
        # )
        # test_dataset.predictions = model.predict(
        #    cell_line_input=cl_features, drug_input=drug_features
        # )
        # TODO final_metric_fold = evaluate(test_dataset)
