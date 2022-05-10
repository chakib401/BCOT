from sklearn import metrics
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari


def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat


def clustering_accuracy(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


def pmi(df, positive=True):
  col_totals = df.sum(axis=0)
  total = col_totals.sum()
  row_totals = df.sum(axis=1)
  expected = np.outer(row_totals, col_totals) / total
  df = df / expected
  # Silence distracting warnings about log(0):
  with np.errstate(divide='ignore'):
    df = np.log(df)
  df[~np.isfinite(df)] = 0.0  # log(0) = 0
  if positive:
    df[df < 0] = 0.0
  return df  


def average_pmi_per_cluster(x, labels):
  values = 0
  pmi_mat = pmi(x @ x.T, positive=False)
  for c in np.unique(labels):
    intra = pmi_mat[labels == c][:, labels == c]
    inter = pmi_mat[labels == c][:, labels != c]
    v = np.mean(intra) - np.mean(inter)
    values += v * np.sum(labels == c) / len(labels)
  return values

