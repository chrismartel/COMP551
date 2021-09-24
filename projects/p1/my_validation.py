# Authors: Christian Martel, Luka Loignon, Marie Guertin
# Date: 2021-09-20

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time


# VALIDATION METRICS

# define MSE loss function
loss = lambda y, y_pred: np.mean((y-y_pred)**2)

#define accuracy function
error_rate = lambda y, y_pred: np.sum(y != y_pred)/y.size



# CROSS-VALIDATION HELPER FUNCTIONS

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


def knn_cross_validation(x, y, K_list, L=10, validation_metric_fn=error_rate):
    '''Perform L-Fold cross-validation on x for each K (number of neighbor) value in K_list using 
       scikit-learn KNN model.
       
       x: features
       y: true prediction labels
       K_list: list of values for parameter K
       L: number of folds
       validation_metric_fn: function used to validate the prediction
    '''
    num_instances = x.shape[0]
        
    # l-fold cross validation splits
    n_folds_splits = cross_validate_splits(num_instances, n_folds=L)
    
    if int(1/L*num_instances) < max(K_list):
        print("Error: max number of neighbors is bigger than validation set size")

    # K x L matrix
    val_matrix = np.zeros((len(K_list),L))
    train_matrix = np.zeros((len(K_list),L))

    for i, K in enumerate(K_list):
        model = KNeighborsClassifier(n_neighbors=K)
        for j, split in enumerate(n_folds_splits):
            # split into train set and validation set
            x_train, x_val = x[split[0], :], x[split[1], :]
            y_train, y_val = y[split[0]], y[split[1]]

            model.fit(x_train, y_train)

            y_pred_val = model.predict(x_val)
            y_pred_train = model.predict(x_train)
            
            val_metric = validation_metric_fn(y_val, y_pred_val)            
            train_metric = validation_metric_fn(y_train, y_pred_train)

            val_matrix[i][j] = val_metric
            train_matrix[i][j] = train_metric

    return val_matrix, train_matrix

# define MSE loss function
loss = lambda y, y_pred: np.mean((y-y_pred)**2)

#define accuracy function
accuracy = lambda y, y_pred: np.sum(y == y_pred)/y.size


# For Decision Tree with Maximum Depth Parameter

def dt_cross_validation_max_depth(x, y, maximum_depth, L=10, validation_metric_fn=error_rate):
    '''x: features
       y: true prediction labels
       maximum_depth: list of values for chosen parameter 
       L: number of folds
       validation_metric_fn: function used to validate the prediction
    '''
    num_instances = x.shape[0]
        
    # l-fold cross validation splits
    n_folds_splits = cross_validate_splits(num_instances, n_folds=L)
    
    if int(1/L*num_instances) < max(maximum_depth):
        print("Error: max number of neighbors is bigger than validation set size")

    # K x L matrix
    val_matrix = np.zeros((len(maximum_depth),L))
    train_matrix = np.zeros((len(maximum_depth),L))

    for i, K in enumerate(maximum_depth):
        model = DecisionTreeClassifier(max_depth=K)
        for j, split in enumerate(n_folds_splits):
            # split into train set and validation set
            x_train, x_val = x[split[0], :], x[split[1], :]
            y_train, y_val = y[split[0]], y[split[1]]

            model.fit(x_train, y_train)

            y_pred_val = model.predict(x_val)
            y_pred_train = model.predict(x_train)
            
            val_metric = validation_metric_fn(y_val, y_pred_val)            
            train_metric = validation_metric_fn(y_train, y_pred_train)

            val_matrix[i][j] = val_metric
            train_matrix[i][j] = train_metric

    return val_matrix, train_matrix

# For Decision Tree with Minimum Samples per Leaf Parameter

def dt_cross_validation_min_samples(x, y, minimum_samples, L=10, validation_metric_fn=error_rate):
    '''x: features
       y: true prediction labels
       minimum_samples: list of values for chosen parameter 
       L: number of folds
       validation_metric_fn: function used to validate the prediction
    '''
    num_instances = x.shape[0]
        
    # l-fold cross validation splits
    n_folds_splits = cross_validate_splits(num_instances, n_folds=L)
    
    if int(1/L*num_instances) < max(minimum_samples):
        print("Error: min numbers of samples per leaf is bigger than validation set size")

    # K x L matrix
    val_matrix = np.zeros((len(minimum_samples),L))
    train_matrix = np.zeros((len(minimum_samples),L))

    for i, K in enumerate(minimum_samples):
        model = DecisionTreeClassifier(min_samples_leaf=K)
        for j, split in enumerate(n_folds_splits):
            # split into train set and validation set
            x_train, x_val = x[split[0], :], x[split[1], :]
            y_train, y_val = y[split[0]], y[split[1]]

            model.fit(x_train, y_train)

            y_pred_val = model.predict(x_val)
            y_pred_train = model.predict(x_train)
            
            val_metric = validation_metric_fn(y_val, y_pred_val)            
            train_metric = validation_metric_fn(y_train, y_pred_train)

            val_matrix[i][j] = val_metric
            train_matrix[i][j] = train_metric

    return val_matrix, train_matrix

# For max depth param and min samples per leaf param combined

def decision_tree_cross_validation(x,y,max_d_list,min_samples_per_leaves_list,L,validation_metric_fn=error_rate):

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
                
                y_pred_val = model.predict(x_val)
                y_pred_train = model.predict(x_train)

                val_metric = validation_metric_fn(y_val, y_pred_val)
                train_metric = validation_metric_fn(y_train, y_pred_train)

                val_matrix[i][j] = val_metric
                train_matrix[i][j] = train_metric

    return val_matrix, train_matrix