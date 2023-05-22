import os.path


def set_environment(experiment_name, root_folder, raw_file_name, random_state=0):
    environment = {
        "raw_input_path_file": os.path.join(root_folder, "data", "raw", raw_file_name),
        "interim_output_path_file": os.path.join(root_folder, "data", "interim", experiment_name,
                                                            "data.csv"),
        "processed_output_path_file": os.path.join(root_folder, "data", "processed", experiment_name,
                                                              "data.csv"),
        "ready_output_path_file": os.path.join(root_folder, "data", "ready", experiment_name, "data.pkl"),
        "save_model_in": os.path.join(root_folder, "models", experiment_name, "best_model.pkl"),
        "save_figures_in": os.path.join(root_folder, "reports", "figures", experiment_name),

        ## Make Dataset Configurations
        "index_col": "No",
        "columns_to_drop_from_our_knowledge": ['Selfratedsensitivitytaste', 'Selfratedsensitivitysmell',
                                          'Selfratedsensitivityspicyness', 'Selfratedsensitivityastringency'],
        "duplicated_columns": ['sour_low_int2', 'sweet_low_int2', 'bitter_low_int2', 'salty_low_int2', 'sour_high_int2',
                          'salty_high_int2', 'bitter_high_int2', 'sweet_high_int2', 'astring_low_int2',
                          'astring_high_int2', 'hot_low_int2', 'hot_high_int2', 'score_low_high_int2',
                          'score_low_high2', 'score_lowhigh_capsadstr_int2',
                          'sour_low_taste2_correct', 'sweet_low_taste2_correct', 'bitter_low_taste2_correct',
                          'salty_low_taste2_correct', 'sour_high_taste2_correct', 'salty_high_taste2_correct',
                          'bitter_high_taste2_correct', 'sweet_high_taste2_correct', 'astring_low_taste2_correct',
                          'astring_high_taste2_correct', 'hot_low_taste2_correct', 'hot_high_taste2_correct',
                          'score_lowhigh_capsadstr2', 'PTC2'],

        "columns_have_lot_missing_values": ['Beginningofdisease', 'mesi'],
        "columns_to_drop_from_data_analysis": ['Satisfactiontastein', 'spicyfoodmealspermonth', 'PTC1'],
        "correlated_columns_to_drop": ['score_low_high_int1', 'score_low_high1'],
        "columns_to_ignore": ['SCORE_LOW_HIGH_CAPSADSTRIN_SOMMA'],
        "categorical_columns": ['sex_f1_m2'],

        # Training Configurations
        "number_folds": 3,
        "target_column_name": "score_lowhigh_capsadstr_int1",
        "param_grid": {
                        'bootstrap': [True],
                        'max_depth': [2, 4, 6],
                        'max_features': [1, 3, 5],
                        'min_samples_leaf': [1],
                        'min_samples_split': [2, 4],
                        'n_estimators': [50, 100, 200],
                        'criterion': ["squared_error"],
                        'random_state': [random_state]
        }
    }

    return environment

