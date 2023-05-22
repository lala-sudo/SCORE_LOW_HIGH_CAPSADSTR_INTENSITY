import os
import ast
import click
import logging
from pathlib import Path

import numpy as np

random_seed = 0
np.random.seed(random_seed)

from src.config import set_environment
from src.data.make_dataset import make_dataset
from src.features.build_features import build_features
from src.models.split_data import split_data
from src.models.train_model import get_best_model
from src.models.train_model import evaluate
from src.visualization.visualize import save_results

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


@click.command()
@click.option('--experiment_name', type=str)
@click.option('--root_folder', type=click.Path(exists=True))
@click.option('--raw_file_name', type=click.Path())
@click.option("--filter_by", type=str)
def main(experiment_name, root_folder, raw_file_name, filter_by):
    logging.info("0. Setting Environment")
    filter_by = dict(ast.literal_eval(filter_by))
    env = set_environment(experiment_name=experiment_name,
                          root_folder=root_folder,
                          raw_file_name=raw_file_name)

    columns_to_not_consider = env["columns_to_drop_from_our_knowledge"] + env["duplicated_columns"] + \
                              env["columns_have_lot_missing_values"] + env["columns_to_drop_from_data_analysis"] + \
                              env["correlated_columns_to_drop"] + env["columns_to_ignore"]

    logging.info("1. Making Dataset")
    if not os.path.exists(env["interim_output_path_file"]):
        logging.info(f"1.1 Interim file in {env['interim_output_path_file']} Does Not Exists, Creating New One")
        make_dataset(env["raw_input_path_file"],
                     index_col=env["index_col"],
                     drop_all_rows_with_na=True,
                     columns_to_not_consider=columns_to_not_consider,
                     filter_by=filter_by,
                     save_in=env["interim_output_path_file"])

    logging.info("2. Building Features")
    if not os.path.exists(env["processed_output_path_file"]):
        build_features(df=env['interim_output_path_file'],
                       index_col=env["index_col"],
                       categorical_columns=env["categorical_columns"],
                       save_in=env["processed_output_path_file"])

    logging.info("3. Split Data")
    if not os.path.exists(env["ready_output_path_file"]):
        split_data(df=env["processed_output_path_file"],
                   index_col=env["index_col"],
                   target_column_name=env["target_column_name"],
                   number_folds=env["number_folds"],
                   random_seed=random_seed,
                   save_in=env["ready_output_path_file"])

    logging.info("4. Get Best Model")
    grid_search, split_data_, X, y = get_best_model(data_split_path_file=env["ready_output_path_file"],
                                                    param_grid=env["param_grid"],
                                                    save_in=env["save_model_in"],
                                                    random_seed=random_seed)
    metrics = evaluate(split_data_, grid_search.best_estimator_)
    logging.info(f"4.1 Metrics of best model: {metrics}")

    logging.info("5. Save Results")
    is_done = save_results(save_model_in=env["save_model_in"],
                           save_figures_in=env["save_figures_in"],
                           metrics=metrics,
                           random_seed=random_seed)

    logging.info(f"Results are saved with status {is_done}")


if __name__ == '__main__':
    project_dir = Path(__file__).resolve()
    logging.info(f"Running {project_dir}")
    main()
