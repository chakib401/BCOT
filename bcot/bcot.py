def bcot(cost, k, w=None, v=None, r=None, c=None, algorithm='emd', n_iter=100, reg=1): 
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


def average_cosine_per_cluster(x, labels):
  from sklearn.metrics.pairwise import cosine_similarity
  values = 0
  for c in np.unique(labels):
    y = x[labels == c]
    v = cosine_similarity(y)
    v = np.mean(v)
    values += v*np.sum(labels == c)/len(labels)
  return values


def average_pmi_per_cluster(x, labels):
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

  values = 0
  pmi_mat = pmi(x @ x.T, positive=False)
  for c in np.unique(labels):
    intra = pmi_mat[labels == c][:, labels == c]
    inter = pmi_mat[labels == c][:, labels != c]
    v = np.mean(intra) - np.mean(inter)
    values += v * np.sum(labels == c) / len(labels)
  return values
