import mysklearn.myutils as myutils
import random
import math
import copy

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        random.seed(random_state)
    
    if shuffle: 
        [X, y] = myutils.parallel_shuffle([X, y])
        
    if not myutils.is_int(test_size):
        test_size = math.ceil(len(X) * test_size)
    
    split_index = len(X) - test_size
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_train_folds = []
    X_test_folds = []
    
    # We're going to make a list of the split-up folds to start.
    n = 0
    folds = []
    for i in range(n_splits):
        folds.append([])
    
    # Just get the indices
    while n < len(X):
        folds[n % n_splits].append(n)
        n += 1
        
    # Now we need to use these folds to generate n_splits sets for each
    for i in range(n_splits):
        new_train_fold = []
        new_test_fold = []
        for j in range(n_splits):
            if i == j:
                for instance in folds[j]:
                    new_test_fold.append(copy.deepcopy(instance))
            else:
                for instance in folds[j]:
                    new_train_fold.append(copy.deepcopy(instance))
                    
        X_train_folds.append(copy.deepcopy(new_train_fold))
        X_test_folds.append(copy.deepcopy(new_test_fold))

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # We can use a ton of the previous function's contents
    X_train_folds = []
    X_test_folds = []
        
    # THIS part is different. We need to...
    # ... group them together.
    # It's easier if we make a new table for this case
    grouped_table = []
    for i in range(len(X)):
        copy_row = []
        copy_row.append(i)
        copy_row.append(y[i])
        grouped_table.append(copy_row)
        
    header = list(range(len(grouped_table)))
    grouped_table, index_folds = myutils.group(grouped_table, header, len(grouped_table[0])-1, include_indices=True)
    
    folds = []
    for i in range(n_splits):
        folds.append([])
    
    # ... make new folds
    n = 0
    for indices in index_folds:
        i = 0
        while i < len(indices):
            folds[i % n_splits].append(index_folds[n][i])
            i += 1
        n += 1
        
    # Now we need to use these folds to generate n_splits sets for each
    for i in range(n_splits):
        new_train_fold = []
        new_test_fold = []
        for j in range(n_splits):
            if i == j:
                for instance in folds[j]:
                    new_test_fold.append(copy.deepcopy(instance))
            else:
                for instance in folds[j]:
                    new_train_fold.append(copy.deepcopy(instance))
                    
        X_train_folds.append(copy.deepcopy(new_train_fold))
        X_test_folds.append(copy.deepcopy(new_test_fold))

    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    for i in range(len(labels)):
        new_row = []
        for i in range(len(labels)):
            new_row.append(0)
        matrix.append(copy.deepcopy(new_row))
    
    for i in range(len(y_true)):
        matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1
    
    return matrix
