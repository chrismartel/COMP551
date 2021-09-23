///cross validation
# Authors: Christian Martel, Luka Loignon, Marie Guertin
# Date: 2021-09-20

from sklearn.tree import DecisionTreeClassifier
import numpy as np


# VALIDATION METRICS

# define MSE loss function
loss = lambda y, y_pred: np.mean((y-y_pred)**2)

#define accuracy function
accuracy = lambda y, y_pred: np.sum(y == y_pred)/y.size

def cross_validate_splits(n, n_folds=5):
    '''Generate splitted validation/train index groups '''
    splits = []
    n_val = n // n_folds
    inds = np.random.permutation(n)
    inds = []
    for f in range(n_folds):
        tr_inds = []
        # get the validation indexes
        val_inds = list(range(f * n_val, (f+1) * n_val))
        
        # get the training indexes
        tr_inds = list()
        if f < n_folds - 1:
            tr_inds = list(range((f+1) * n_val, n))
        
        if f > 0:
            tr_inds = tr_inds + list(range(0, f * n_val))
        splits.append((tr_inds, val_inds))
    return splits

def decision_tree_cross_validation(x,y,max_d_list,min_samples_per_leaves_list,L,validation_metric_fn):
    
    num_instances = x.shape[0]
    
    # l-fold cross validation splits
    n_folds_splits = cross_validate_splits(num_instances, n_folds=L)

    # validation metrics
    val_matrix = np.zeros((len(max_d_list),len(min_samples_per_leaves_list),L))
    train_matrix = np.zeros((len(max_d_list),len(min_samples_per_leaves_list),L)) 

    for i, M in enumerate(max_d_list):
        for j, S in enumerate(min_samples_per_leaves_list):
            model = DecisionTreeClassifier(max_depth = max_d_list[i], min_samples_leaf = min_samples_per_leaves_list[j])
            for k, split in enumerate(n_folds_splits):
                # split into train set and validation set
                x_train, x_val = x[split[0], :], x[split[1], :]
                y_train, y_val = y[split[0]], y[split[1]]

                model.fit(x_train, y_train)
                                                  
                val_metric  = validation_metric_fn(y_val, y_pred_val)  
                train_metric  = validation_metric_fn(y_train, y_pred_train)

                val_matrix[i][j] = val_metric
                train_matrix[i][j] = train_metric
                                                  
    return val_matrix, train_matrix