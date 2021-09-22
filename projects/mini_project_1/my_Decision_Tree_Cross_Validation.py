def decision_tree_cross_validation(x,y,max_d_list,min_samples_per_leaves_list,L):

    # l-fold cross validation splits
    n_folds_splits = cross_validate_splits(num_instances, n_folds=L)

    # validation metrics
    err_matrix, err_val_matrix  = np.zeros((len(max_d_list),len(min_samples_per_leaves_list),L)),np.zeros((len(max_d_list),len(min_samples_per_leaves_list),L))
    err_train_matrix = np.zeros((len(max_d_list),len(min_samples_per_leaves_list),L))

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

                err_val = loss(y_val, y_pred_val)
                err_train = loss(y_train, y_pred_train)
                                                  
                err_val_matrix [i][j][k] = err_val
                err_train_matrix[i][j][k] = err_train
                                                  
    return err_val_matrix, err_train_matrix