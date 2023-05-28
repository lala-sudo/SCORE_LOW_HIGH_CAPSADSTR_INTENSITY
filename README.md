# Score Low High Capsadstr
The analysis of taste function of healthy controls and patients.

## About:

This project holds the technical materials used to realize the results shared in our research project: `A supervised Learning regression method for the analysis of taste function of healthy controls (HC) and patients with chemosensory loss. `

By leveraging the code and models within this repository, you can re-run the machine-learning models, conduct experiments, or even use them as a starting point for developing your own models. The source code can be found in the **src** folder. There where is defined the `main.py` file that runs all the project pipeline steps:

1. Make Dataset: In this step, we read raw data and handle missing values.
2. Build Features: In this step, we encode categorical features.
3.  Data Analysis: Create the correlation matrix for each subset (when filter data is applied) and manually conduct Fisher Test using Statistica.  ⚠️ The correlation matrix and Fisher Test shared in our work are created manually from all the datasets, without filtering by controls or patients.
4. Create Cross Validation folds: Split data into different folds.
5. Train Machine Learning Models: Find the best model and hyperparameters, combing grid search and cross-validation.
6. Plot figures and Save Final Results: Create model Explainability AI and residual figures.

## ⛳️ Project Structure:

```
├── configurations: Configurations and Parameters of all experiments
├── data: Folder where is stores all data
│   ├── interim
│   │   ├── 20230524
│   │   │   ├── main_experiment_controls
│   │   │   ├── main_experiment_patients
│   │   │   ├── second_experiment_controls
│   │   │   ├── second_experiment_patients
│   │   │   ├── third_experiment_controls
│   │   │   └── third_experiment_patients
│   │
│   ├── processed
│   │   ├── 20230524
│   │   │   ├── main_experiment_controls
│   │   │   ├── main_experiment_patients
│   │   │   ├── second_experiment_controls
│   │   │   ├── second_experiment_patients
│   │   │   ├── third_experiment_controls
│   │   │   └── third_experiment_patients
│   │   
│   │       
│   ├── raw
│   └── ready
│       │   
│       └── 20230524
│           ├── main_experiment_controls
│           ├── main_experiment_patients
│           ├── second_experiment_controls
│           ├── second_experiment_patients
│           ├── third_experiment_controls
│           └── third_experiment_patients
├── models: Folder where is stored binary Machine Learning
│   │   
│   └── 20230524
│       ├── main_experiment_controls
│       ├── main_experiment_patients
│       ├── second_experiment_controls
│       ├── second_experiment_patients
│       ├── third_experiment_controls
│       └── third_experiment_patients
├── notebooks: Notebooks folders
├── references
├── reports: Folder where is stores figures
│   ├── figures
│   │   └── 20230524
│   │       ├── main_experiment_controls
│   │       │   └── shap_values
│   │       ├── main_experiment_patients
│   │       │   └── shap_values
│   │       ├── second_experiment_controls
│   │       ├── second_experiment_patients
│   │       ├── third_experiment_controls
│   │       └── third_experiment_patients
│   └── profiles
│       └── 20230524
│           ├── main_experiment_controls
│           ├── main_experiment_patients
│           ├── second_experiment_controls
│           ├── second_experiment_patients
│           ├── third_experiment_controls
│           └── third_experiment_patients
└── src: Source code Folder
   ├── __init__.py
   ├── config.py: Read Configurations 
   ├── data
   │   ├── __init__.py
   │   └── make_dataset.py: Read DataSet
   ├── features
   │   ├── __init__.py
   │   ├── analyze_data.py: Create Correlation Matrix
   │   └── build_features.py: Encode Features
   ├── main.py: Main Pipiline
   ├── models
   │   ├── __init__.py
   │   ├── split_data.py: Create Cross-validation folds
   │   └── train_model.py: Train the best model
   └── visualization
       ├── __init__.py
       └── visualize.py: Plot figures and save results


```

## 📘 Setup Your Environment

1. Install Python: [3.7.9](https://www.python.org/downloads/release/python-379/)
2. Install the virtual environment: (Via **Terminal Interface**)
   1. Install virtualenv: `pip install virtualenv`
   2. Create your virtual environment: `python3 -m venv research_py_379 `(You can choose another name)
   3. Update pip by: `python3 -m pip install --upgrade pip`
   4. Activate your virtual environment: `source research_py_379/bin/activate`
3. Clone or Download the project from this repository.
4. In the terminal where the environment is active SCORE_LOW_HIGH_CAPSADSTR_INTENSITY: go to the root folder of the cloned project.
5. Install required Python libraries `pip install -e .`

## 🚦 Run Different Experiments

From the root folder in **Terminal** where you cloned the project (make sure that your virtual environment is activated), you can run the following commands to run the experiments:

- [Option 1] Main Experiment using only Controls Data:`python src/main.py --experiment_name "20230524" --configuration_file_name "main_experiment_controls.json" --force_model_rebuild True`
- [Option 2] Main Experiment using only Patients' Data:`python src/main.py --experiment_name "20230524" --configuration_file_name "main_experiment_patients.json" --force_model_rebuild True`

♻️ You can specify different experiment_name if you want to re-force data building steps. \
♻️ You can set force_model_rebuild to False if you want to skip models re-training for the same experiment_name.

***

Identically, you can run the Secondary experiments:

- [Secondary] The second experiment used only Controls Data and the two most important features:`python src/main.py --experiment_name "20230524" --configuration_file_name "second_experiment_controls.json"`
- [Secondary] The second experiment used only Patients' Data and the two most important features:`python src/main.py --experiment_name "20230524" --configuration_file_name "second_experiment_patients.json"`
- [Secondary] The third experiment used only Controls Data and the most important feature:`python src/main.py --experiment_name "20230524" --configuration_file_name "third_experiment_controls.json"`
- [Secondary] The third experiment used only Patients' Data and the most important feature:`python src/main.py --experiment_name "20230524" --configuration_file_name "third_experiment_patients.json"`

## 💬 We're here to help!

- Contact us: lala.chaimae.naciri@gmail.com