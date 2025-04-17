import numpy as np
import pandas as pd
import shap


def get_vips(x, model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)

    df_importance = pd.DataFrame({
        'Scores': vips
    }, index=x.columns)

    df_sorted = df_importance.sort_values(by='Scores', ascending=False)
    return df_sorted


def get_shap(x, model):
    explainer = shap.Explainer(model)
    shap_values = explainer(x)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    df_importance = pd.DataFrame({
        'Scores': mean_abs_shap
    }, index=x.columns)
    df_sorted = df_importance.sort_values(by='Scores', ascending=False)
    return df_sorted
