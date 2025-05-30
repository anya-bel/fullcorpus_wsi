import bcubed
import numpy as np
from sklearn import metrics
from sklearn.metrics import v_measure_score, adjusted_rand_score, normalized_mutual_info_score


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def inv_purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)


def f_purity_score(y_true, y_pred):
    purity = purity_score(y_true, y_pred)
    inv_purity = inv_purity_score(y_true, y_pred)
    return (2 * purity * inv_purity / (purity + inv_purity))


def evaluate_clusters(true, pred):
    purity = purity_score(true, pred)
    inv_purity = inv_purity_score(true, pred)
    f_pur = f_purity_score(true, pred)
    ari = adjusted_rand_score(true, pred)
    v_mes = v_measure_score(true, pred)
    nmi = normalized_mutual_info_score(true, pred, average_method='max')

    ldict = {f"item{i}": set([elem]) for i, elem in enumerate(true)}
    cdict = {f"item{i}": set([elem]) for i, elem in enumerate(pred)}
    b_precision = bcubed.precision(cdict, ldict)
    b_recall = bcubed.recall(cdict, ldict)
    f_bcubed = bcubed.fscore(b_precision, b_recall)
    metrics = {
        'Purity': purity,
        'Inverse purity': inv_purity,
        'F-purity': f_pur,
        'ARI': ari,
        'V-measure': v_mes,
        'B-Cubed Precision': b_precision,
        'B-Cubed Recall': b_recall,
        'F-B-Cubed': f_bcubed,
        'NMI': nmi,
    }

    return metrics
