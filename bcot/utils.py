import numpy as np
import os.path
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer



def binarize(a, k, sparse=True):
  b = np.zeros((len(a), k))
  b[np.arange(len(a)), a] = 1
  return b

def sinkhorn_scaling(a, b, cost):
  u = np.ones(len(a))
  v = np.ones(len(b))
  cost = cost + 1e-7
  u_prev = u
  v_prev = v

  for _ in range(100):
    v = b / (cost.T @ u)
    # v = np.nan_to_num(v, nan=0., posinf=0., neginf=0.)
    u = a / (cost @ v)
    if not np.isfinite(u).all() or not np.isfinite(v).all(): 
      u = u_prev
      v = v_prev
      break
    # u = np.nan_to_num(u, nan=0., posinf=0., neginf=0.)
    if np.abs(u-u_prev).mean() < 1e-5: break
    u_prev = u
  
  return u[:, None] * cost * v


def read_dataset(dataset, sparse=False):
    data = sio.loadmat(os.path.join('data', f'{dataset}.mat'))
    features = data['fea'].astype(float)
    if not sp.issparse(features):
      features = sp.csr_matrix(features)
    labels = data['gnd'].reshape(-1) - 1
    n_classes = len(np.unique(labels))
    return features, labels, n_classes


