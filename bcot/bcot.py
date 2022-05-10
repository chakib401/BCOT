import numpy as np
import ot
from bcot.utils import sinkhorn_scaling

def BCOT(cost, k, w=None, v=None, r=None, c=None, algorithm='emd', n_iter=100, reg=1): 
  if w is None: w = np.ones(cost.shape[0]) / cost.shape[0]
  if v is None: v = np.ones(cost.shape[1]) / cost.shape[1]
  if r is None: r = np.ones(k) / k
  if c is None: c = np.ones(k) / k

  Z = np.random.randint(k, size=(cost.shape[0], k))
  if algorithm == 'emd': 
    Z = ot.emd(w, r, Z,  numItermax=int(1e7))
  if algorithm == 'sinkhorn':
    Z = ot.bregman.sinkhorn(w, r, Z, reg=reg,  numItermax=int(1e7), warn=False)
  if algorithm == 'sinkhorn+':
    Z = sinkhorn_scaling(w, r, Z)

  loss_old = np.inf

  for _ in range(n_iter):
    if algorithm == 'emd':
      W = ot.emd(v, c, cost.T @ Z, numItermax=int(1e7))
      Z = ot.emd(w, r, cost @ W, numItermax=int(1e7))
    if algorithm == 'sinkhorn':
      P = cost.T @ Z
      W = ot.sinkhorn(v, c, P, reg=reg, numItermax=int(1e7), warn=False)
      P = cost @ W
      Z = ot.sinkhorn(w, r, P, reg=reg, numItermax=int(1e7), warn=False)
    if algorithm == 'sinkhorn+':
      W = sinkhorn_scaling(v, c, cost.T @ Z)
      Z = sinkhorn_scaling(w, r, cost @ W)

    if _ % 10 == 0:
      if type(cost) == np.ndarray:
        loss = np.sum(cost * (Z @ W.T))
      else:
        loss = np.sum(cost.multiply(Z @ W.T))

      if np.abs(loss_old - loss) <  1e-7: break
      loss_old = loss

  return Z, W

