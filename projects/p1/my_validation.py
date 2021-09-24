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
    train_set_size = int((L-1)/L*num_instances);
        
    # l-fold cross validation splits
    n_folds_splits = cross_validate_splits(num_instances, n_folds=L)
    
    # K x L matrix
    val_matrix = np.zeros((len(K_list),L))
    train_matrix = np.zeros((len(K_list),L))


    for i, K in enumerate(K_list):
        if K > train_set_size:
          break
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
def dt_cross_validation(x, y, max_depth_list=list(), min_samples_per_leaf_list=list(), L=10, validation_metric_fn=error_rate):
    '''x: features
       y: true prediction labels
       max_depth_list: range of values to test for max depth
       min_samples_per_leaf_list: range of values to test for min_samples_per_leaf
       L: number of folds
       validation_metric_fn: function used to validate the prediction
    '''
    num_instances = x.shape[0]

    # l-fold cross validation splits
    n_folds_splits = cross_validate_splits(num_instances, n_folds=L)
    
    # parameter ranges
    max_depth_range = 1 if not max_depth_list else len(max_depth_list)
    min_samples_range = 1 if not min_samples_per_leaf_list else len(min_samples_per_leaf_list)

    # K x L matrix
    val_matrix, train_matrix = np.zeros((max_depth_range,min_samples_range,L)), np.zeros((max_depth_range,min_samples_range,L))

    for i in range(max_depth_range):
      md = None if not max_depth_list else max_depth_list[i]
      for j in range(min_samples_range):
          ms = 1 if not min_samples_per_leaf_list else min_samples_per_leaf_list[j]
          model = DecisionTreeClassifier(max_depth=md, min_samples_leaf=ms)
          for k, split in enumerate(n_folds_splits):
              # split into train set and validation set
              x_train, x_val = x[split[0], :], x[split[1], :]
              y_train, y_val = y[split[0]], y[split[1]]

              model.fit(x_train, y_train)

              y_pred_val, y_pred_train = model.predict(x_val), model.predict(x_train)
          
              val_metric, train_metric = validation_metric_fn(y_val, y_pred_val), validation_metric_fn(y_train, y_pred_train)           

              val_matrix[i][j][k], train_matrix[i][j][k] = val_metric, train_metric

    return np.squeeze(val_matrix), np.squeeze(train_matrix)