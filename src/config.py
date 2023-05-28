import os
import json


def set_environment(experiment_name, root_folder, configuration_file_name):
    configuration_file_path = os.path.join(root_folder, 'configurations', configuration_file_name)
    with open(configuration_file_path, 'r') as f:
        configurations = json.load(f)

    configuration_name = configuration_file_name.split(".")[0]

    settings = {
        "raw_input_path_file": os.path.join(root_folder, "data", "raw", configurations["data_source"]),
        "interim_output_path_file": os.path.join(root_folder, "data", "interim",
                                                 experiment_name, configuration_name,
                                                 "data.csv"),
        "processed_output_path_file": os.path.join(root_folder, "data", "processed",
                                                   experiment_name, configuration_name,
                                                   "data.csv"),
        "analyzed_output_path_folder": os.path.join(root_folder, "reports", "profiles",
                                                    experiment_name, configuration_name),
        "ready_output_path_file": os.path.join(root_folder, "data", "ready",
                                               experiment_name, configuration_name,
                                               "data.pkl"),
        "save_model_in": os.path.join(root_folder, "models",
                                      experiment_name, configuration_name,
                                      "best_model.pkl"),
        "save_metrics_in": os.path.join(root_folder, "models",
                                        experiment_name, configuration_name,
                                        "metrics.json"),
        "save_figures_in": os.path.join(root_folder, "reports", "figures",
                                        experiment_name, configuration_name)
    }

    columns_to_not_consider = configurations["data"]["columns_to_drop_from_our_knowledge"] + \
                              configurations["data"]["duplicated_columns"] + \
                              configurations["data"]["columns_have_lot_missing_values"] + \
                              configurations["data"]["columns_to_drop_from_data_analysis"] + \
                              configurations["data"]["correlated_columns_to_drop"] + \
                              configurations["data"]["columns_to_ignore"]

    settings['columns_to_not_consider'] = columns_to_not_consider
    settings['index_col'] = configurations['data']['index_col']
    settings['categorical_columns_to_encode'] = configurations['data']['categorical_columns_to_encode']
    settings['categorical_columns_encoded'] = configurations['data']['categorical_columns_encoded']
    settings['filter_by'] = configurations['data']['filter_by']
    settings['train'] = configurations['train']

    return settings
