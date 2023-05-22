import os
import json
import pickle

import numpy as np
import pandas as pd

import shap
import matplotlib.pyplot as plt


def calculate_feature_importance(shap_values, column_names):
    # Calculate absolute mean SHAP values for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)

    # Normalize feature importance values
    feature_importance = mean_shap_values / np.sum(mean_shap_values)

    # Calculate percentage for each feature value
    total_percentage = np.sum(feature_importance)
    percentage_values = np.round((feature_importance / total_percentage) * 100, 2)

    # Create a DataFrame with columns: "Feature", "Importance", "Percentage"
    df = pd.DataFrame({
        "Feature": column_names,
        "Importance": feature_importance,
        "Percentage": percentage_values
    })

    # Sort the DataFrame in descending order of feature importance
    df = df.sort_values(by="Importance", ascending=False)

    return df.reset_index(drop=True).copy()


def save_results(save_model_in, save_figures_in, summary_plots=False, risiduals_plot=False, metrics=None, save_ranking=False, random_seed=0):
    np.random.seed(random_seed)
    directory = os.path.dirname(save_model_in)

    with open(save_model_in, 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(directory, "X.pkl"), 'rb') as f:
        X = pickle.load(f)

    with open(os.path.join(directory, "y.pkl"), 'rb') as f:
        y = pickle.load(f)

    y_pred = model.predict(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if save_figures_in is not None:
        if not os.path.exists(save_figures_in):
            os.makedirs(save_figures_in)

        if summary_plots is True:
            shap.initjs()
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.savefig(os.path.join(save_figures_in, "features_importance_plot.jpg"),
                        dpi=1000,
                        bbox_inches='tight')

            plt.cla()
            shap.summary_plot(shap_values, X, show=False)
            plt.savefig(os.path.join(save_figures_in, "summary_plot.jpg"),
                        dpi=1000,
                        bbox_inches='tight')

        if metrics is not None:
            mean_ = {}
            for metric_name in metrics[list(metrics.keys())[0]].keys():
                mean_[metric_name] = np.mean([metrics[fold][metric_name] for fold in metrics.keys()])

            metrics["mean"] = mean_

            with open(os.path.join(save_figures_in, "metrics.json"), 'w') as fp:
                json.dump(metrics, fp)

        if risiduals_plot is True:
            plt.cla()
            plt.figure(figsize=(12, 6))
            plt.scatter(x=y, y=y_pred, c="#7CAE00", alpha=0.5)
            z = np.polyfit(y, y_pred, 1)
            p = np.poly1d(z)
            plt.plot(y, p(y), "#F8766D")
            plt.ylabel('Predicted intensity score')
            plt.xlabel('Experimental intensity score')
            plt.xticks(size=18)
            plt.yticks(size=18)
            plt.savefig(os.path.join(save_figures_in, "residuals.jpg"),
                        dpi=1000,
                        bbox_inches='tight')

        if save_ranking is True:
            features_ranking = calculate_feature_importance(shap_values, X.columns)
            features_ranking.to_csv(os.path.join(save_figures_in, "features_ranking.csv"))

    return True
