# SCORE_LOW_HIGH_CAPSADSTR_INTENSITY
Regression model to predict the overall taste perception in whole subject as a first experiment, in healthy subjects in the second experiment, and for patients with chemosensory loss in the third experiment


# Run

'pip install -e .'
'
 python src/main.py --root_folder "/Users/lalachaimaenaciri/PycharmProjects/SCORE_LOW_HIGH_CAPSADSTR_INTENSITY" --raw_file_name "transcelerator mar 2022_explanation Melania.xlsx" --filter_by "{'Group_cotnrol1_patient2': '*'}" --experiment_name "exp_id_your_datetime_all_subjects"
 python src/main.py --root_folder "/Users/lalachaimaenaciri/PycharmProjects/SCORE_LOW_HIGH_CAPSADSTR_INTENSITY" --raw_file_name "transcelerator mar 2022_explanation Melania.xlsx" --filter_by "{'Group_cotnrol1_patient2': 1}" --experiment_name "exp_id_your_datetime_controls"
 python src/main.py --root_folder "/Users/lalachaimaenaciri/PycharmProjects/SCORE_LOW_HIGH_CAPSADSTR_INTENSITY" --raw_file_name "transcelerator mar 2022_explanation Melania.xlsx" --filter_by "{'Group_cotnrol1_patient2': 2}" --experiment_name "exp_id_your_datetime_patients"

'