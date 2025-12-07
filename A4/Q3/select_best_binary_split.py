import numpy as np

from tree_utils import LeafNode, InternalDecisionNode

def select_best_binary_split(x_NF, y_N, MIN_SAMPLES_LEAF=1):
    ''' Determine best single feature binary split for provided dataset

    Args
    ----
    x_NF : 2D array, shape (N,F) = (n_examples, n_features)
        Training data features at current node we wish to find a split for.
    y_N : 1D array, shape (N,) = (n_examples,)
        Training labels at current node.
    min_samples_leaf : int
        Minimum number of samples allowed at any leaf.

    Returns
    -------
    feat_id : int or None, one of {0, 1, 2, .... F-1}
        Indicates which feature in provided x array is used for best split.
        If None, a binary split that improves the cost is not possible.
    thresh_val : float or None
        Value of x[feat_id] at which we threshold.
        If None, a binary split that improves the cost is not possible.
    x_LF : 2D array, shape (L, F)
        Training data features assigned to left child using best split.
    y_L : 1D array, shape (L,)
        Training labels assigned to left child using best split.
    x_RF : 2D array, shape (R, F)
        Training data features assigned to right child using best split.
    y_R : 1D array, shape (R,)
        Training labels assigned to right child using best split.

    Examples
    --------
    # Example 1a: Simple example with F=1 and sorted features input
    >>> N = 6
    >>> F = 1
    >>> x_NF = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((6, 1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    0
    >>> thresh_val
    2.5

    # Example 1b: Same as 1a but just scramble the order of x
    # Should give same results as 1a
    >>> x_NF = np.asarray([2.0, 1.0, 0.0, 3.0, 5.0, 4.0]).reshape((6, 1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    0
    >>> thresh_val
    2.5

    # Example 2: Advanced example with F=12 total features
    # Fill the features such that middle column is same as 1a above,
    # but the first 6 columns with random features
    # and the last 6 columns with all zeros
    >>> N = 6
    >>> F = 13
    >>> prng = np.random.RandomState(0)
    >>> x_N1 = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((6,1))
    >>> x_NF = np.hstack([prng.randn(N, F//2), x_N1, np.zeros((N, F//2))])
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    6
    >>> thresh_val
    2.5

    # Example 3: binary split isn't possible (because all x same)
    >>> N = 5
    >>> F = 1
    >>> x_NF = np.asarray([3.0, 3.0, 3.0, 3.0, 3.0]).reshape((5,1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id is None
    True

    # Example 4: binary split isn't possible (because all y same)
    >>> N = 5
    >>> F = 3
    >>> prng = np.random.RandomState(0)
    >>> x_NF = prng.rand(N, F)
    >>> y_N  = 1.2345 * np.ones(N)
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id is None
    True
    '''
    N, F = x_NF.shape

    # Allocate space for best split info per feature
    cost_F = np.inf * np.ones(F)
    thresh_val_F = np.zeros(F)
    
    for f in range(F):
        # Get all unique threshold candidates
        xunique_U = np.unique(x_NF[:, f])
        possib_xthresh_V = 0.5 * (xunique_U[:-1] + xunique_U[1:])
        V = possib_xthresh_V.size
        
        # Filter thresholds to respect MIN_SAMPLES_LEAF
        if MIN_SAMPLES_LEAF > 1:
            m = MIN_SAMPLES_LEAF - 1
            possib_xthresh_V = possib_xthresh_V[m:(V-m)]
        V = possib_xthresh_V.size
        
        if V == 0:
            cost_F[f] = np.inf
            continue
        
        # Evaluate cost for each threshold candidate
        left_yhat_V = np.zeros(V)
        right_yhat_V = np.zeros(V)
        left_cost_V = np.zeros(V)
        right_cost_V = np.zeros(V)
        
        for v in range(V):
            thresh = possib_xthresh_V[v]
            
            # Split data based on threshold
            left_mask = x_NF[:, f] < thresh
            right_mask = ~left_mask
            
            y_left = y_N[left_mask]
            y_right = y_N[right_mask]
            
            # Compute predictions (mean) and costs (MSE)
            left_yhat_V[v] = np.mean(y_left)
            right_yhat_V[v] = np.mean(y_right)
            
            left_cost_V[v] = np.sum(np.square(y_left - left_yhat_V[v]))
            right_cost_V[v] = np.sum(np.square(y_right - right_yhat_V[v]))
        
        total_cost_V = left_cost_V + right_cost_V
        
        # Check if any split improves the model
        costs_all_the_same = np.allclose(total_cost_V, total_cost_V[0])
        yhat_all_the_same = np.allclose(left_yhat_V, right_yhat_V)
        
        if costs_all_the_same and yhat_all_the_same:
            cost_F[f] = np.inf
            continue
        
        # Select threshold with minimum cost
        chosen_v_id = np.argmin(total_cost_V)
        cost_F[f] = total_cost_V[chosen_v_id]
        thresh_val_F[f] = possib_xthresh_V[chosen_v_id]
    
    # Find best feature to split on
    best_feat_id = np.argmin(cost_F)
    best_thresh_val = thresh_val_F[best_feat_id]
    
    if not np.isfinite(cost_F[best_feat_id]):
        return (None, None, None, None, None, None)
    
    # Create left and right child datasets
    left_mask_N = x_NF[:, best_feat_id] < best_thresh_val
    right_mask_N = ~left_mask_N
    x_LF, y_L = x_NF[left_mask_N], y_N[left_mask_N]
    x_RF, y_R = x_NF[right_mask_N], y_N[right_mask_N]
    
    return (best_feat_id, best_thresh_val, x_LF, y_L, x_RF, y_R)
