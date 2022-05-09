from sklearn.cluster import KMeans
import scipy.sparse as ss
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
import ot
from gcc.metrics import clustering_accuracy
from sklearn.utils.extmath import randomized_svd
from gcc.utils import read_dataset, preprocess_dataset
from time import time

np.seterr(divide='ignore')


def binarize(a, k, sparse=True):
  b = np.zeros((len(a), k))
  b[np.arange(len(a)), a] = 1
  return b

def sinkhorn_scaling(a, b, cost):
  u = np.ones(len(a))
  v = np.ones(len(b))
  # cost = cost + 1e-7
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
