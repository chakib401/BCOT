import numpy as np

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
