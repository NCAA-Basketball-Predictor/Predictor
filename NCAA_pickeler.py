import pickle # standard python library
import importlib
import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils
import mysklearn.myclassifiers
importlib.reload(mysklearn.myclassifiers)
from mysklearn.myclassifiers import MyRandomForestClassifier
import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation
import copy
import random

# "pickle" an object (AKA object serialization)
# save a Python object to a binary file

# "unpickle" an object (AKA object de-serialization)
# load a Python object from a binary file (back into memory)

# for your project, pickle an instance MyRandomForestClassifier, MyDecisionTreeClassifier
# for demo use header and interview_tree below
header, data = myutils.load_from_file("input_data/NCAA_Statistics_24444.csv")
random.seed(15)

# Now, we can move to create some decision trees. Let's first create trees over the whole dataset, then
# test upon our stratisfied k-fold splitting method.

class_col = myutils.get_column(data, header, "Win Percentage")
data = myutils.drop_column(data, header, "Win Percentage")
data = myutils.drop_column(data, header, "Scoring Margin")
atts = header[1:-1]

X_indices = range(len(class_col))
X_train_folds, X_test_folds = myevaluation.stratified_kfold_cross_validation(X_indices, class_col, n_splits=10)
my_rf = MyRandomForestClassifier()

for fold_index in range(len(X_train_folds)):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    for train_index in X_train_folds[fold_index]:
        X_train.append(copy.deepcopy(data[train_index]))
        y_train.append(copy.deepcopy(class_col[train_index]))
        
    for test_index in X_test_folds[fold_index]:
        X_test.append(copy.deepcopy(data[test_index]))
        y_test.append(copy.deepcopy(class_col[test_index]))
        
    # Get a classifier in here...

# Fitting...
    my_rf.fit(X_train, y_train, n_trees=50, m_trees=10, min_atts=2)

packaged_object = [header, my_rf.trees]
# pickle packaged_object
outfile = open("best_classifier.p", "wb")
pickle.dump(packaged_object, outfile)
outfile.close()