import mysklearn.myutils as myutils
import copy
import cmath
import math
import random
import os, sys

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        # 1. First, we'll get the means of xs and ys
        x_bar = sum(X_train) / len(X_train)
        y_bar = sum(y_train) / len(y_train)
    
        # 2. Now we'll use loops to simulate sigmas in the equation
        num = 0
        den = 0

        for i in range(len(X_train)):
            num += (X_train[i] - x_bar) * (y_train[i] - y_bar)
            den += (X_train[i] - x_bar) ** 2

        m = num / den

        b = y_bar - (m * x_bar)

        self.slope = m
        self.intercept = b

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        return [self.slope * x + self.intercept for x in X_test]

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """        
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        indices = []
        
        # NOTE: I had to upend this implementation for PA5 given the number of merge_sorts we were running.
        # This should be quicker. It's more quadratic, at least.
        n = self.n_neighbors
        
        # We're going to do two different things given different types of data. Namely,
        # IF all the data is categorical, we'll use a different strategy.
        # Here is that strategy in action:
        categorical = True
        for att in X_test[0]:
            if not isinstance(att, str):
                categorical = False
                break
                
        if categorical:
            # For each test...
            for test in X_test:
                # Create an att+1 list of empty lists
                indices_by_distances = []
                for _ in range(len(X_test[0])+1):
                    indices_by_distances.append([])
                
                # For each instance in the training set...
                for i in range(len(self.X_train)):
                    # Compute the distance by counting the number of differences.
                    distance = 0
                    for att in range(len(test)):
                        if test[att] != self.X_train[i][att]:
                            distance += 1
                    
                    # Notice we don't take the sqrt, we do that later for posterity
                    # Add it to the indices list
                    indices_by_distances[distance].append(i)
                
                # Now we start grabbing indices.
                # We expect the user to have set the seed beforehand
                k_distances = []
                k_indices = []
                
                distance = 0
                while len(k_indices) < self.n_neighbors:
                    if len(indices_by_distances[distance]) == 0:
                        distance += 1
                    else:
                        k_distances.append(distance)
                        get_index = random.randrange(0, len(indices_by_distances[distance]))
                        k_indices.append(indices_by_distances[distance][get_index])
                        indices_by_distances[distance].pop(get_index)
                        
                # Finally, we append it to the whole data
                distances.append(copy.deepcopy(k_distances))
                indices.append(copy.deepcopy(k_indices))
        
        # Otherwise, we know it's continuous in at least one respect. I'll feel better about speed and etc.
        else:
            for test in X_test:
                # Create a parallel list of our closest numbers. Starting point is untoppable.
                k_distances = [len(test) + 1 for _ in range(n)]
                k_indices = [-1 for _ in range(n)]

                # For each instance in X_train...
                for i in range(len(self.X_train)):
                    # Compute the distance
                    distance = 0
                    for att in range(len(self.X_train[i])):
                        # Check for categorical data...
                        if isinstance(test[att], str):
                            if test[att] != self.X_train[i][att]:
                                distance += 1
                        else:
                            distance += (test[att] - self.X_train[i][att]) ** 2
                    distance **= 0.5

                    # Check if/where it should go in our storage...
                    storage_index = 0
                    placed = False

                    while not placed and storage_index < n:
                        if distance < k_distances[storage_index]:
                            k_distances.insert(storage_index, distance)
                            k_distances = k_distances[:-1]
                            k_indices.insert(storage_index, i)
                            k_indices = k_indices[:-1]
                            placed = True
                        else:
                            storage_index += 1

                distances.append(copy.deepcopy(k_distances))
                indices.append(copy.deepcopy(k_indices))
            
        return distances, indices
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        
        # We'll use our previous implementation. 
        distances, indices = self.kneighbors(X_test)
        
        # Now we need to gather all the available unique class labels as given by y_train.
        unique_classes = myutils.get_all_unique_values(self.y_train)
        for i in range(len(X_test)):
            classes_count = []
            
            for classification in unique_classes:
                classes_count.append(0)
            
            # Count each classification
            for index in indices[i]:
                for j in range(len(unique_classes)):
                    if self.y_train[index] == unique_classes[j]:
                        classes_count[j] += 1
            
            # Going to use the closest point to break ties
            most_classes = []
            most_count = 0
            
            for j in range(len(classes_count)):
                if classes_count[j] > most_count:
                    most_classes.clear()
                    most_count = classes_count[j]
                    most_classes.append(unique_classes[j])
                elif classes_count[j] == most_count:
                    most_classes.append(unique_classes[j])
                    
            class_to_add = most_classes[0]
            if len(most_classes) > 1:
                tieb = 0
                found = False
                while found is False:
                    if self.y_train[indices[i][tieb]] in most_classes:
                        found = True
                        class_to_add = self.y_train[indices[i][tieb]]
                    else:
                        tieb += 1
                # This should never run out of indices.
                
            y_predicted.append(copy.deepcopy(class_to_add))
            
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(list of list of 2-tuple of int): The prior probabilities computed for each
            label in the training set. First in 2-tuple is numerator, second is denominator.
        posteriors(list of list of list of 2-tuple of int): The posterior probabilities computed for each
            attribute value/label pair in the training set. First in 2-tuple is numerator, second is den.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train
        
        if len(X_train) == 0:
            pass
        
        # 1. Group a copy of X_train by y_test
        # Even though we have a myutils grouping function, I'd rather do this manually.
        uniques = myutils.get_all_unique_values(y_train)
        myutils.merge_sort(uniques)
        
        # After getting all the unique classes alphabetically, we group by them, not including our classes
        grouped_table = []
        for unique in uniques:
            new_table = []
            for i in range(len(X_train)):
                if y_train[i] == unique:
                    new_table.append(copy.deepcopy(X_train[i]))
            grouped_table.append(new_table)
        
        # 2. Now that we've got our grouped table, we can get to work.
        # For each group, we need a prior.
        self.priors = []
        total_num = len(y_train)
        
        for unique in uniques:
            prior = [0, total_num]
            for instance in y_train:
                if instance == unique:
                    prior[0] += 1
            self.priors.append(copy.deepcopy(prior))
            
        # 3. Now for the posteriors, a much tougher process.
        # First, we want to get the number of unique values for each attribute in all instances.
        attribute_names = []
        for i in range(len(X_train[0])):
            new_row = []
            att_column = myutils.get_all_unique_values(myutils.get_column(X_train, None, i))
            myutils.merge_sort(att_column)
            attribute_names.append(copy.deepcopy(att_column))
        
        # We want to group by each unique classification, so we'll outer loop on that
        self.posteriors = []
        for u in range(len(uniques)):
            total_in_group = len(grouped_table[u])
            # We'll separate the next level by attribute...
            new_outer_list = []
            for i in range(len(attribute_names)):
                # And finally by attribute designation
                new_inner_list = []
                my_att_column = myutils.get_column(grouped_table[u], None, i)
                for att in attribute_names[i]:
                    new_rational = [0, total_in_group]
                    for item in my_att_column:
                        if item == att:
                            new_rational[0] += 1
                    new_inner_list.append(copy.deepcopy(new_rational))
                
                new_outer_list.append(copy.deepcopy(new_inner_list))
                
            self.posteriors.append(copy.deepcopy(new_outer_list))
            
        # Should be good.

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # 1. First, we need to generate the lists that store the attribute labels without considering priors
        attribute_names = []
        for i in range(len(self.X_train[0])):
            new_row = []
            att_column = myutils.get_all_unique_values(myutils.get_column(self.X_train, None, i))
            myutils.merge_sort(att_column)
            attribute_names.append(copy.deepcopy(att_column))
            
        # We also need the sorted unique classification list
        uniques = myutils.get_all_unique_values(self.y_train)
        myutils.merge_sort(uniques)
        # This should be the same length as the number of tables in posteriors:
        assert len(uniques) == len(self.posteriors)
        
        # Now that we have this...
        # 2. For each test instance in X_test...
        y_test = []
        for test in X_test:
            # 3. We operate for each unique class value given
            class_chances = []
            for c in range(len(uniques)):
                new_rational = [1, 1]
                
                # 4. We gather the multiplied rational
                # We can start with the priors
                new_rational[0] *= self.priors[c][0]
                new_rational[1] *= self.priors[c][1]
                
                # And now we move to posteriors...
                for a in range(len(attribute_names)):
                    # Early exit...
                    if new_rational[0] == 0:
                        break
                        
                    # Get the index of the appropriate classification... if it's there
                    if test[a] in attribute_names[a]:
                        index = attribute_names[a].index(test[a])
                        # And multiply appropriately!
                        new_rational[0] *= self.posteriors[c][a][index][0]
                        new_rational[1] *= self.posteriors[c][a][index][1]
                    else:
                        # If we can't find it, we have a case where we have an unseen attribute in the
                        # training set. We're going to handle this a bit wierdly; We're going to ignore
                        # the attribute, instead multiplying both the numerator and denominator by its
                        # prior's number of instances. This is for a few reasons; It helps us build a very
                        # easy tiebreaker, and we don't lose all our hard work. Let's do it.
                        new_rational[0] *= self.priors[c][0]
                        new_rational[1] *= self.priors[c][0]
                    
                # Now we add it...
                class_chances.append(copy.deepcopy(new_rational))
            
            # 5. Evaluate the class_chances and predict it!
            classes = copy.deepcopy(uniques)
            # Convert rationals to floats
            chances = []
            for chance in class_chances:
                chances.append(myutils.convert_rational_to_float(chance))
                
            max_indices = myutils.get_max_indices(chances)
            
            # 6. Check for ties...
            if len(max_indices) > 1:
                # ... and tiebreak as follows:
                # 1) Biggest prior (we can tell by the denominator!)
                # 2) Random number
                denominators = []
                for max_index in max_indices:
                    denominators.append(class_chances[max_index][1])
                    
                max_indices = myutils.get_max_indices(denominators)
                
                if len(max_indices) > 1:
                    # Random selection time!
                    # We assume the user seeds beforehand.
                    max_indices = [random.randrange(0, len(max_indices))]
                    
            y_test.append(classes[max_indices[0]])
        
        return y_test
    
class MyZeroRClassifier:
    """Represents a Zero-R classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        class (obj): The classification to always report

    Notes:
        Really bad. Or, at least, mostly.
    """
    def __init__(self):
        """Initializer for MyRandomClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.my_class = None

    def fit(self, X_train, y_train):
        """Fits a Zero-R classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            This is the really bad part.
        """
        self.X_train = X_train
        self.y_train = y_train
        
        self.my_class = myutils.get_mode(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            self.my_class
            
        Notes:
            Oh my goodness, this hurts to look at
        """
        return [self.my_class for _ in X_test]
    
class MyRandomClassifier:
    """Represents a Random classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        class (obj): The classification to always report

    Notes:
        Better, at least. I think. More difficult.
    """
    def __init__(self):
        """Initializer for MyRandomClassifier.

        """
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a Zero-R classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Trivial.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_pred (list of obj): The predicted classifications
            
        Notes:
            User is expected to seed appropriately beforehand.
        """
        y_pred = []
        for test in X_test:
            index = random.randrange(0, len(self.y_train))
            y_pred.append(self.y_train[index])
            
        return y_pred

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        assert len(X_train) > 0
        assert len(X_train) == len(y_train)
        
        self.X_train = X_train
        self.y_train = y_train
        
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        
        # Definition of select_attribute:
        def select_attribute(current_insts, available_atts):
            """ The function for selecting an attribute.
                Currently based on entropic selection.
                
            Args:
                current_insts: The current table we are working on
                available_atts: The attributes we have available to us
                
            Returns:
                str: The attribute to split on.
            """
            # The random implementation
            # rand_index = random.randrange(0, len(available_atts))
            # return available_atts[rand_index]
            
            # The entropic implementation
            num_classes = len(myutils.get_all_unique_values(myutils.get_column(current_insts, None, len(current_insts[0])-1)))
            entropies = []
            
            # For each attribute available...
            for att in available_atts:
                p = myutils.group_dict(current_insts, available_atts, att)
                
                new_ent = 0
                # For each key in the partition... (Values)
                for key in p:
                    # The denominator: The total number of that value
                    d = len(p[key])
                    
                    # There's a few steps to get the numerator but it's not too bad
                    class_col = myutils.get_column(p[key], None, len(p[key][0])-1)
                    uniques = myutils.get_all_unique_values(class_col)
                    for unique_class in uniques:
                        # Getting the numerator and iterating
                        n = myutils.count_in_list(class_col, count_for=unique_class)
                        # We'll use the weight here
                        w = d/len(current_insts)
                        new_ent -= w * n/d * math.log2(n/d)
                
                entropies.append(copy.deepcopy(new_ent))
                    
            # Now we have the entropies. Let's find the minimum.
            # We'll use the first attribute as the tiebreaker.
            min_ent = 2 # Higher than any possible
            min_att = -1
            for e in range(len(entropies)):
                if entropies[e] < min_ent:
                    min_ent = entropies[e]
                    min_att = e
                    
            return available_atts[min_att]
        
        # Partitioning helper:
        def partition(current_insts, available_atts, split_att, att_domains):
            """ The partitioning helper. Uses the myutils function
            
            Args:
                current_insts (list of lists): The current table to partition
                available_atts (list of str): The header, essentially
                split_att (str): The attribute to partition on
                att_domains (dict (str -> list of str)): The attribute domains
                
            Returns:
                dict (str->list of list): The dictionary that was split on
                list of str: The header with the removed attribute
            """
            d = myutils.group_dict(current_insts, available_atts, split_att, uniques=att_domains[split_att])
            available_atts.remove(split_att)
            return d, available_atts
        
        def check_all_same_class(current_insts):
            """ Checks if all the instances have all the same class.
            
            Args:
                current_insts (list of lists): The current table
                
            Returns:
                bool: The sameness of the whole table
            """
            if len(current_insts) == 0:
                return False
            
            first_class = current_insts[0][-1]
            for row in current_insts:
                if row[-1] != first_class:
                    return False
            
            # If we got here, we're good
            return True
        
        def resolve_clash(current_insts):
            """ What happens upon a clash. Returns the classification.
            
            Args:
                current_insts (list of list): The current table.
                
            Returns:
                str: The classification.
            """
            class_column = myutils.get_column(current_insts, None, len(current_insts[0])-1)
            modes = myutils.get_modes(class_column)
            
            # We need to check for ties here
            if len(class_column):
                return modes[0]
            else:
                # We'll flip a coin/die; Not much else to do.
                return modes[random.randrange(0, len(modes))]
        
        # Definition of tdidt:
        def tdidt(current_insts, available_atts, att_domains, prev_num=None):
            """ The recursive TDIDT method. Defined in scope.
            
            Args:
                current_insts: The current table we are working on
                available_atts: The attributes we have available to us
                att_domains: The attribute domains, in a dict
                prev_num: The previous number of instances in the set.
                          Useful for Case 3, recursive only.
                
            Returns:
                list: The decision tree starting at this point
            """
            # Step 0: Get the number of instances total
            total_num = len(current_insts)
            if prev_num is None:
                prev_num = len(current_insts)
            
            # Step 1: Find the attribute to split on
            split_att = select_attribute(current_insts, available_atts)
            tree = ["Attribute", split_att]
            
            # Step 2: Partition
            partitioned, available_atts = partition(current_insts, available_atts, split_att, att_domains)
            
            # For each key in the dictionary...
            for key in partitioned:
                # We need to add a Value list to the tree for this instance
                new_value_node = ["Value", key]
                
                # Step 3: Base Cases
                # Now we take care of the base cases, else we repeat
                # CASE 1: All class labels are the same
                if check_all_same_class(partitioned[key]):
                    # Make a leaf node
                    new_leaf_node = ["Leaf", partitioned[key][0][-1], len(partitioned[key]), total_num]
                    new_value_node.append(new_leaf_node)
                # CASE 2: Clash on lack of attributes to select
                elif len(partitioned[key]) > 0 and len(available_atts) == 0:
                    # Assess the clash
                    new_leaf_node = ["Leaf", resolve_clash(partitioned[key]), len(partitioned[key]), total_num]
                    new_value_node.append(new_leaf_node)
                # CASE 3: No more instances.
                elif len(partitioned[key]) == 0:
                    # We overwrite the current tree as only a leaf node, unfortunately.
                    # We'll use a return to get out of the loop
                    tree = ["Leaf", resolve_clash(current_insts), total_num, prev_num] # <- Where we need prev_num
                    return tree # Early exit
                # CASE 0: Recurse
                else:
                    subtree = tdidt(partitioned[key], available_atts.copy(), att_domains, total_num)
                    new_value_node.append(subtree)
                
                # After this is done we append the new node to the tree
                tree.append(new_value_node)
                
            return tree
        
        # The initial call to tdidt:
        header = ["att" + str(i) for i in range(len(X_train[0]))]
        available_atts = header.copy()
        
        # We need to get the attribute domains
        att_domains = {}
        for att in available_atts:
            att_domains[att] = myutils.get_all_unique_values(myutils.get_column(train, available_atts, att))
            
        self.tree = tdidt(train, available_atts, att_domains)
        # We're done once the recursion ends
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        header = ["att" + str(i) for i in range(len(self.X_train[0]))]
        
        # We'll define a recursive function
        def rec_predict(tree, test):
            # HELPER
            
            # BASE CASE: Leaf node
            if tree[0] == "Leaf":
                return tree[1]
            # RECURSIVE CASE: Branch node
            else: # "Attribute"
                for att in header:
                    if tree[1] == att: # We gotta get the correct attribute
                        index = header.index(att)
                        # We'll find the correct Value node; The for loop is cut short
                        for i in range(2, len(tree)):
                            value_node = tree[i]
                            # If the attribute value matches.../
                            if value_node[1] == test[index]:
                                # Recurse
                                return rec_predict(value_node[2], test)
                            
                        # If we haven't found it we get a Value Error
                        raise ValueError
                            
        y_pred = []
        for test in X_test:
            y_pred.append(copy.deepcopy(rec_predict(self.tree, test)))
            
        return y_pred

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        # We want a recursive function here. Here's our strategy:
        # - Build up a list of attribute and values
        # - When we reach a leaf node, print the list formatted appropriately
        
        # First, if we didn't get an attribute_names list, we should make it
        if attribute_names is None:
            attribute_names = ["att" + str(i) for i in range(len(self.X_train[0]))]
        
        # Let's do it!
        def rec_print_decision_rules(tree, rules):
            """ Recursive helper for the above."""
            # Base case: Leaf node
            if tree[0] == "Leaf":
                # We want to print out all the rules, joined
                # With "THEN class = label"
                print(" ".join(rules), "THEN", class_name, "=", str(tree[1]))
            # Recursive case: Attribute node
            else: # tree[0] == "Attribute":
                # We need to recurse. This is a bit complex.
                # First we add an "AND" if the length so far is greater than 1
                if len(rules) > 1:
                    rules.append("AND")
                
                # Then we grab the attribute name and add it to the rules
                rules.append(attribute_names[int(tree[1][-1])])
                rules.append("==")
                # Now we do some recursion...
                # For each value in the tree...
                for value_node in tree[2:]:
                    # This recusive part is a bit tricky; We have to be careful here
                    rules.append(value_node[1])
                    # Recurse
                    rec_print_decision_rules(value_node[2], rules.copy())
                    # Backtrack, delete
                    rules = rules[:-1]
                    
                # Now we have to remove some stuff out of the rules
                rules = rules[:-3]
                
        rec_print_decision_rules(self.tree, ["IF"])

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method. # TODO
        """
        # First, if we didn't get an attribute_names list, we should make it
        if attribute_names is None:
            attribute_names = ["att" + str(i) for i in range(len(self.X_train[0]))]
            
        # Now for a similar recursive function to the above. We'll use a list of strs that we'll separate by endls
        # for to writing the DOT file
        def rec_generate_graphviz(tree, lines, attribute_names, atts=[]):
            """ Helper for visualize_tree.
            
            Args:
                tree (tree): The current tree we're upon.
                lines (list of str): The list of strings to add to the text file. PbR
                attribute_names (list of str): The attribute names.
                atts (list of str): Stores labels for attributes. PbR
                
            Returns:
                index (int): If an attribute node, the att# to find the label
                
            Pass by reference for lines.
            """
            # Base Case: Leaf node
            if tree[0] == "Leaf":
                # Add a new label to atts; We do this every time
                atts.append(tree[1])
                # Make a new attribute based on the length of atts
                return_index = len(atts) - 1
                line = "    att" + str(len(atts) - 1)
                # We need to label the node appropriately though
                # It defaults to a bubble node
                line += " [label=\"%s\"];" % str(atts[-1])
                lines.append(copy.deepcopy(line))
            
            # We don't do anything more in the base case
            # Recursive Case: Attribute node
            else:
                # Add a new label to atts; We do this every time
                atts.append(tree[1])
                # Make a new attribute based on the length of atts
                return_index = len(atts) - 1
                line = "    att" + str(return_index)
                # We need to label the node appropriately though
                # It defaults to a bubble node
                line += " [label=\"%s\" shape=box];" % attribute_names[int(atts[return_index][3:])]
                lines.append(copy.deepcopy(line))
                
                # Now, for each value node...
                for value_node in tree[2:]:
                    # ... we do some tricky recursion.
                    index = rec_generate_graphviz(value_node[2], lines, attribute_names, atts)
                    
                    # Now we backtrack and add stuff after
                    # Yuck
                    line = "    att" + str(return_index) + "--att" + str(index) + " [label=\"%s\"];" % value_node[1]
                    lines.append(copy.deepcopy(line))
                    
            return return_index
        
        lines = ["graph g {"]
        rec_generate_graphviz(self.tree, lines, attribute_names)
        lines.append("}")
        
        # Putting it all together...
        with open(dot_fname, 'w') as df:
            df.write("\n".join(lines))
            
        # Command line shenanigans for dot->pdf
        cmd = "dot -Tpdf -o %s %s" % (pdf_fname, dot_fname)
        os.popen(cmd)
