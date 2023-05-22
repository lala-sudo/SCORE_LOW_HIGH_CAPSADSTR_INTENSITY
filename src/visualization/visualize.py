import os
import json
import pickle

import numpy as np

import shap
import matplotlib.pyplot as plt


def save_results(save_model_in, save_figures_in, metrics, random_seed=0):
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

        mean_ = {}
        for metric_name in metrics[list(metrics.keys())[0]].keys():
            mean_[metric_name] = np.mean([metrics[fold][metric_name] for fold in metrics.keys()])

        metrics["mean"] = mean_

        with open(os.path.join(save_figures_in, "metrics.json"), 'w') as fp:
            json.dump(metrics, fp)

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

    return True
