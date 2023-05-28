import os
import click
import logging

from src.config import set_environment
from src.data.make_dataset import make_dataset
from src.features.build_features import build_features
from src.features.analyze_data import analyze_data
from src.models.split_data import split_data
from src.models.train_model import get_best_model
from src.models.train_model import evaluate
from src.models.train_model import save_metrics
from src.visualization.visualize import save_results

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


@click.command()
@click.option('--experiment_name', type=str, required=True)
@click.option("--configuration_file_name", type=str, required=True)
@click.option("--force_model_rebuild", type=bool, required=False)
def main(experiment_name, configuration_file_name, force_model_rebuild=False):
    logging.info("0. Setting Configuration")
    root_folder = os.getcwd()
    logging.info(f"Root Folder: {root_folder}")
    settings = set_environment(experiment_name=experiment_name,
                               configuration_file_name=configuration_file_name,
                               root_folder=root_folder)
    random_seed = settings["train"]["param_grid"]["random_state"][0]

    logging.info("1. Making Dataset ...")
    if not os.path.exists(settings["interim_output_path_file"]) is True:
        logging.info(f"\t1.1 Interim file in {settings['interim_output_path_file']} Does Not Exists, Creating New One")
        dataframe = make_dataset(settings["raw_input_path_file"],
                                 index_col=settings["index_col"],
                                 drop_all_rows_with_na=True,
                                 columns_to_not_consider=settings["columns_to_not_consider"],
                                 filter_by=settings["filter_by"],
                                 save_in=settings["interim_output_path_file"])
        logging.info(f"\t1.2 Make Dataset is Done considering only : {list(dataframe.columns)}")

    logging.info("2. Building Features ...")
    if not os.path.exists(settings["processed_output_path_file"]):
        build_features(df=settings["interim_output_path_file"],
                       index_col=settings["index_col"],
                       categorical_columns=settings["categorical_columns"],
                       save_in=settings["processed_output_path_file"])
        logging.info("\t2.1 Build Features is Done.")

    logging.info("3. Data Analyzing ...")
    if not os.path.exists(settings["analyzed_output_path_folder"]):
        analyze_data(df=settings["processed_output_path_file"],
                     index_col=settings["index_col"],
                     categorical_columns=settings["categorical_columns"],
                     save_in=settings["analyzed_output_path_folder"])
        logging.info("\t3.1 Data Analysis is Done.")

    logging.info("4. Splitting Data ...")
    if not os.path.exists(settings["ready_output_path_file"]):
        split_data(df=settings["processed_output_path_file"],
                   index_col=settings["index_col"],
                   target_column_name=settings["train"]["target_column_name"],
                   number_folds=settings["train"]["number_folds"],
                   random_seed=random_seed,
                   save_in=settings["ready_output_path_file"])
        logging.info("\t4.1 Data Split is Done.")

    logging.info("5. Getting Best Model ...")
    if not os.path.exists(settings["save_model_in"]) or force_model_rebuild is True:
        grid_search, split_data_, X, y = get_best_model(data_split_path_file=settings["ready_output_path_file"],
                                                        param_grid=settings["train"]["param_grid"],
                                                        save_in=settings["save_model_in"],
                                                        random_seed=random_seed)
        metrics = evaluate(split_data_, grid_search.best_estimator_)
        metrics_with_mean = save_metrics(metrics=metrics, save_in=settings["save_metrics_in"])
        logging.info(f"\t5.1 MAPE of best model: {metrics_with_mean['mean']['MAPE']}")

    logging.info("6. Saving Results ...")
    is_done = save_results(save_model_in=settings["save_model_in"],
                           save_figures_in=settings["save_figures_in"],
                           summary_plots=True,
                           residuals_plot=True,
                           save_ranking=True,
                           random_seed=random_seed)
    logging.info(f"✅ Results are saved with status {is_done}")

    return True


if __name__ == '__main__':
    main()
