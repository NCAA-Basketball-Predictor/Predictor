# Unit Tests for knn and decision trees (find testing for RF's in random_forest.ipynb):

import numpy as np
import scipy.stats as stats
import random
import math

from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDecisionTreeClassifier
import mysklearn.myutils as myutils

four_training_samples_training_data = [
    [7,7],
    [7,4],
    [3,4],
    [1,4],
    [3,7]
]
train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6],
        [2, 3]
    ]
train_labels = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
book_train = [
    [0.8,6.3],
    [1.4,8.1],
    [2.1,7.4],
    [2.6,14.3],
    [6.8,12.6],
    [8.8,9.8],
    [9.2,11.6],
    [10.8,9.6],
    [11.8,9.9],
    [12.4,6.5],
    [12.8,1.1],
    [14.0,19.9],
    [14.2,18.5],
    [15.6,17.4],
    [15.8,12.2],
    [16.6,6.7],
    [17.4,4.5],
    [18.2,6.9],
    [19.0,3.4],
    [19.6,11.1],
    [9.1,11.0]
]
book_labels = [
    "-",
    "-",
    "-",
    "+",
    "-",
    "+",
    "-",
    "+",
    "+",
    "+",
    "-",
    "-",
    "-",
    "-",
    "-",
    "+",
    "+",
    "+",
    "-",
    "+"
]
book_distances = [[9.538343671728336, 
        8.228000972289685, 
        7.87146746166177, 
        7.2897187874430385, 
        2.8017851452243794, 
        1.2369316876852974, 
        0.6082762530298216, 
        2.202271554554525, 
        2.9154759474226513, 
        5.580322571321482, 
        10.56882207249228, 
        10.159724405711012, 
        9.069729874698584, 
        9.121951545584968, 
        6.806614430096655, 
        8.645229898620396, 
        10.54229576515476, 
        9.980981915623332, 
        12.480785231707179, 
        10.500476179678712
    ]]
book_indicies = [
    [6,5,7,4,8]
]

def test_kneighbors_classifier_kneighbors():
    k_neighbors = MyKNeighborsClassifier(n_neighbors=3)
    normalized_training_data = myutils.normalize_data(four_training_samples_training_data)
    y_train = ["bad","bad","good","good"]
    actual_indicies = [[0,2,3]]
    actual_distances = [[0.6666666666666667, 1.2018504251546631, 1, 1.0540925533894598]]
    k_neighbors.fit(normalized_training_data[:-1],y_train)
    distances, neighbor_indicies = k_neighbors.kneighbors(normalized_training_data[-1:])
    assert neighbor_indicies == actual_indicies
    assert np.allclose(distances,actual_distances)
    k_neighbors_1 = MyKNeighborsClassifier(n_neighbors=3)
    normalized_training_data_1 = myutils.normalize_data(train)
    actual_distances_1 = [[0.23570226039551587, 0.8333333333333334, 0.4714045207910317, 0.3726779962499649, 0.23570226039551584, 0.5, 0.3333333333333333, 0.5270462766947299]]
    actual_indicies_1 = [[4,0,6]]
    k_neighbors_1.fit(normalized_training_data_1[:-1],train_labels)
    distances_1, neighbor_indicies_1 = k_neighbors_1.kneighbors(normalized_training_data_1[-1:])
    assert neighbor_indicies_1 == actual_indicies_1
    assert np.allclose(distances_1,actual_distances_1)
    k_neighbors_book = MyKNeighborsClassifier(n_neighbors=5)
    k_neighbors_book.fit(book_train[:-1],book_labels)
    distances_book, neighbor_indicies_book = k_neighbors_book.kneighbors(book_train[-1:])
    assert book_indicies == neighbor_indicies_book
    assert np.allclose(distances_book,book_distances)


def test_kneighbors_classifier_predict():
    k_neighbors = MyKNeighborsClassifier(n_neighbors=3)
    normalized_training_data = myutils.normalize_data(four_training_samples_training_data)
    y_train = ["bad","bad","good","good"]
    k_neighbors.fit(normalized_training_data[:-1],y_train)
    y_prediction = k_neighbors.predict(normalized_training_data[-1:])
    y_actual = ['good']
    assert y_prediction == y_actual
    k_neighbors_1 = MyKNeighborsClassifier(n_neighbors=3)
    normalized_training_data_1 = myutils.normalize_data(train)
    k_neighbors_1.fit(normalized_training_data_1[:-1],train_labels)
    y_1_predicted = k_neighbors_1.predict(normalized_training_data_1[-1:])
    y_1_actual = ['yes']
    assert y_1_predicted == y_1_actual
    k_neighbors_book = MyKNeighborsClassifier(n_neighbors=5)
    k_neighbors_book.fit(book_train[:-1],book_labels)
    book_label_predicted = k_neighbors_book.predict(book_train[-1:])
    book_label_actual = ['+']
    assert book_label_actual == book_label_predicted

    
def test_decision_tree_classifier_fit():
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]
    
    interview_tree = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]
    
    my_dt = MyDecisionTreeClassifier()
    y_train = myutils.get_column(interview_table, interview_header, "interviewed_well")
    interview_table = myutils.drop_column(interview_table, interview_header, "interviewed_well")
    X_train = interview_table
    my_dt.fit(X_train, y_train)
    
    assert myutils.equivalent(my_dt.tree, interview_tree) # Above this function
    
    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

    # Computed using entropy; This won't work until this is implemented
    # This took me an hour, but near the end it got easy. I'm glad computers exist.
    degrees_tree = \
        ["Attribute", "att0",
            ["Value", "A",
                ["Attribute", "att4",
                    ["Value", "A",
                        ["Leaf", "FIRST", 5, 14]
                    ],
                    ["Value", "B",
                        ["Attribute", "att3",
                            ["Value", "A", 
                                ["Attribute", "att1", 
                                    ["Value", "A", 
                                        ["Leaf", "FIRST", 1, 2]
                                    ],
                                    ["Value", "B",
                                        ["Leaf", "SECOND", 1, 2]
                                    ]
                                ]
                            ],
                            ["Value", "B",
                                ["Leaf", "SECOND", 7, 9]
                            ]
                        ]
                    ]
                ]
            ],
            ["Value", "B",
                ["Leaf", "SECOND", 12, 26]
            ]
        ]
    
    # Same thing this time
    my_dt = MyDecisionTreeClassifier()
    y_train = myutils.get_column(degrees_table, degrees_header, "Class")
    degrees_table = myutils.drop_column(degrees_table, degrees_header, "Class")
    X_train = degrees_table
    my_dt.fit(X_train, y_train)
    
    assert myutils.equivalent(my_dt.tree, degrees_tree)

def test_decision_tree_classifier_predict():
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]
    
    my_dt = MyDecisionTreeClassifier()
    y_train = myutils.get_column(interview_table, interview_header, "interviewed_well")
    interview_table = myutils.drop_column(interview_table, interview_header, "interviewed_well")
    X_train = interview_table
    my_dt.fit(X_train, y_train)
    
    X_test = [["Junior", "Java", "yes", "no"],
              ["Junior", "Java", "yes", "yes"]]
    y_test = ["True", "False"]
    
    assert myutils.equivalent(my_dt.predict(X_test), y_test)
    
    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]
    
    my_dt = MyDecisionTreeClassifier()
    y_train = myutils.get_column(degrees_table, degrees_header, "Class")
    degrees_table = myutils.drop_column(degrees_table, degrees_header, "Class")
    X_train = degrees_table
    my_dt.fit(X_train, y_train)
    
    X_test = [
        ["B", "B", "B", "B", "B"],
        ["A", "A", "A", "A", "A"],
        ["A", "A", "A", "A", "B"]
    ]
    y_test = ["SECOND", "FIRST", "FIRST"]
    
    assert myutils.equivalent(my_dt.predict(X_test), y_test)
    
    # After this we can feel pretty darn good about our implementation.
    # Because it was tricky I'm going to run a good few more tests on the back end.
    
def test_decision_tree_classifier_print_rules():
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]
    
    my_dt = MyDecisionTreeClassifier()
    y_train = myutils.get_column(interview_table, interview_header, "interviewed_well")
    interview_table = myutils.drop_column(interview_table, interview_header, "interviewed_well")
    X_train = interview_table
    my_dt.fit(X_train, y_train)
    
    print("Interview Tree Rules:")
    my_dt.print_decision_rules(interview_header[:-1], interview_header[-1])
    print()
    
    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]
    
    my_dt = MyDecisionTreeClassifier()
    y_train = myutils.get_column(degrees_table, degrees_header, "Class")
    degrees_table = myutils.drop_column(degrees_table, degrees_header, "Class")
    X_train = degrees_table
    my_dt.fit(X_train, y_train)
    
    print("Degrees Tree Rules:")
    my_dt.print_decision_rules(degrees_header[:-1], degrees_header[-1])
    print()
    
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]
    
    my_dt = MyDecisionTreeClassifier()
    y_train = myutils.get_column(interview_table, interview_header, "interviewed_well")
    interview_table = myutils.drop_column(interview_table, interview_header, "interviewed_well")
    X_train = interview_table
    my_dt.fit(X_train, y_train)
    
    print("Interview Tree Rules with generic names:")
    my_dt.print_decision_rules()
    print()
    
    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]
    
    my_dt = MyDecisionTreeClassifier()
    y_train = myutils.get_column(degrees_table, degrees_header, "Class")
    degrees_table = myutils.drop_column(degrees_table, degrees_header, "Class")
    X_train = degrees_table
    my_dt.fit(X_train, y_train)
    
    print("Degrees Tree Rules with generic names:")
    my_dt.print_decision_rules()
    print()
    
test_decision_tree_classifier_fit()
test_decision_tree_classifier_predict()
test_decision_tree_classifier_print_rules()


