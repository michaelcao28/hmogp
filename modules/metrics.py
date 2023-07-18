import numpy as np

def scotts_pi(confusion_matrix, weight_type = "unweighted", return_weights=False):

    cm = confusion_matrix
    w = np.zeros_like(cm, dtype=np.float64)
    row, col = cm.shape
    cat = np.arange(1,row+1) # Category starts from 1,2,...K
    
    # Compute the weights
    for i in range(row):
        # j = i
        for j in range(col):
            if i == j:
                w_res = 1
            else:
                if weight_type == "quadratic":
                        w_res = 1 - (cat[i] - cat[j])**2 / (cat.max() - 1)**2
                elif weight_type == "linear":
                    w_res = 1 - np.abs(cat[i] - cat[j]) / (cat.max() - 1)
                elif weight_type == "ordinal":
                    m_ij = max(cat[i], cat[j]) - min(cat[i], cat[j]) + 1
                    m_max = cat.max() - cat.min() + 1
                    w_res = 1 - (m_ij / m_max)
                elif weight_type == "radical":
                    w_res = 1 - (np.sqrt(np.abs(cat[i] - cat[j])) / np.sqrt(np.abs(cat.max() - 1)))
                elif weight_type == "unweighted":
                    w_res = 0
            w[i,j] = w_res
            w[j,i] = w_res
    
    total = cm.sum()
    row_marginal = cm.sum(-1) / total
    col_marginal = cm.sum(0) / total
    freq = cm / total
    P_0 = (w * freq).sum()
    
    # Compute P_e 
    P_e = 0.0
    for i in range(row):
        j = i
        p_i = (row_marginal[i] + col_marginal[i]) / 2
        p_j = (row_marginal[j] + col_marginal[j]) / 2
        P_e += w[i,j] * p_i * p_j

    A = np.round((P_0 - P_e) / (1 - P_e), 4)
    
    if return_weights:
        return w, A
    else:
        return A
    
def clr(x):
    """
    Perform centre log ratio (clr) Aitchison transformation.
    Parameters
    ----------
    x: numpy.ndarray
       A matrix of compositions (rows).  x can be a single
       composition or a 2d array of compositions forming a data set.
    Returns
    -------
    numpy.ndarray
         clr-transformed data projected to R^(n-1).
    """
    assert x.ndim == 1, "Valid for 1D arrays only"
    if np.any(x <= 0):
        raise ValueError("Cannot have negative or zero proportions")
    logx = np.log(x)
    gx = np.mean(logx, axis=-1, keepdims=True)
    return logx - gx

def aitchison_distance(x, y):
    """
    Aitchison distance between two compositions.
    Parameters
    ----------
    x, y: numpy.ndarrays
       Compositions
    Returns
    -------
    numpy.float64
         A real value of this distance metric >= 0.
    """
    assert x.ndim == 1 and y.ndim == 1, "Valid for 1D arrays only"
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError(
            "Cannot have negative or zero proportions - parameter 0.")
    return np.linalg.norm(clr(x / y))

def mean_sum_composition_errors(X, Y):
    """
    Mean Sum of Compositional Errors (MSCE)
    """
    assert X.shape == Y.shape, "X.shape and Y.shape must be equal."
    sce = np.array([aitchison_distance(X[i], Y[i]) for i in range(Y.shape[0])])
    mean_sce = sce.mean()
    return mean_sce
