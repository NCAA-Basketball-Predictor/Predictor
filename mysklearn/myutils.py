import os
import csv
import copy
import random
import numpy as np

def is_int(x):
    '''
    Returns whether or not the string x is an integer type. HELPER
    
    Arguments:
        x (str): The value to check
        
    Returns:
        bool: telling whether or not x is an integer type
        
    Citation:
        Found from https://stackoverflow.com/questions/15357422/python-determine-if-a-string-should-be-converted-into-int-or-float
    '''
    
    try:
        a = float(x)
        b = int(a)
    except(TypeError, ValueError):
        return False
    else:
        return a == b
        
def is_float(x):
    '''
    Returns whether or not the string x is a float type. HELPER
    
    Arguments:
        x (str): The value to check
        
    Returns:
        bool: telling whether or not x is a float type
        
    Citation:
        Found from https://stackoverflow.com/questions/15357422/python-determine-if-a-string-should-be-converted-into-int-or-float
    '''
    
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True
        
def get_column(table, header, column_id, include_missing_values=True):
    """ Extracts a column from the table data as a list.
    
    Args:
        table (list of lists): The table of data to extract from
        header (list of str): The header of that table, giving the column
            names.
        column_id (int or str): If a str, the column name to grab.
            If an int, the column index.
        include_missing_values (bool): True if missing values should be
            included, or False otherwise.
            
    Returns:
        list of obj: 1D list of values in the column found.
    
    Notes:
       Raises ValueError on invalid column_id
    """
    column_num = -1
    # Check if column_id is an integer
    if is_int(column_id):
        if column_id < 0 or (header is not None and column_id >= len(header)): # Out of range
            raise ValueError
        else:
            column_num = column_id
    else: # its a str
        for i in range(len(header)):
            if column_id == header[i]:
                column_num = i
                break
        if column_num == -1: # If unfound
            raise ValueError
    
    # Gets the column
    my_column = []
    for row in table:
        if (row[column_num] != "NA" and row[column_num] != "N/A" \
            and row[column_num] != "") or include_missing_values:
            my_column.append(row[column_num])
    
    return my_column

def drop_column(table, header, column_id):
    """ Drops a column from the table.
    
    Args:
        table (list of lists): The table of data to drop
        header (list of str): The header of that table, giving the column
            names.
        column_id (int or str): If a str, the column name to drop.
            If an int, the column index.
            
    Returns:
        list of obj: The new table
    
    Notes:
       Raises ValueError on invalid column_id
    """
    column_num = -1
    # Check if column_id is an integer
    if is_int(column_id):
        if column_id < 0 or (header is not None and column_id >= len(header)): # Out of range
            raise ValueError
        else:
            column_num = column_id
    else: # its a str
        for i in range(len(header)):
            if column_id == header[i]:
                column_num = i
                break
        if column_num == -1: # If unfound
            raise ValueError
    
    # Gets the column
    new_table = []
    for row in table:
        new_table.append(row[:column_num] + row[column_num+1:])
    
    return new_table

def get_mode(my_list):
    """ Gets the mode in a list. Discrete. Defaults to first alphabetically.
    
    Args:
        my_list (list of obj): The list to find from.
        
    Returns:
        obj: The mode.
    """
    assert len(my_list) > 0
    
    if len(my_list) == 1:
        return my_list[0]
    
    # The "clean way" to do this involves sorting. We want a linear solution instead.
    values = []
    counts = []
    for item in my_list:
        if item in values:
            counts[values.index(item)] += 1
        else:
            values.append(item)
            counts.append(1)
            
    # Find the max index
    max_index = 0
    max_count = counts[0]
    for i in range(1, len(counts)):
        if counts[i] > max_count:
            max_index = i
            max_count = counts[i]
            
    return values[max_index]

def get_modes(my_list):
    """ Gets the mode(s) in a list. Discrete. Gets a list instead.
    
    Args:
        my_list (list of obj): The list to find from.
        
    Returns:
        list: The modes
    """
    assert len(my_list) > 0
    
    if len(my_list) == 1:
        return my_list[0]
    
    # The "clean way" to do this involves sorting. We want a linear solution instead.
    values = []
    counts = []
    for item in my_list:
        if item in values:
            counts[values.index(item)] += 1
        else:
            values.append(item)
            counts.append(1)
            
    # Find the max index
    max_indices = [0]
    max_count = counts[0]
    for i in range(1, len(counts)):
        if counts[i] > max_count:
            max_indices = [i]
            max_count = counts[i]
        elif counts[i] == max_count:
            max_indices.append(i)
            
    return [values[i] for i in max_indices]
    
def get_median(my_list):
    """ Gets the median in a list of numeric values.
    
    Args:
        my_list (list of ordinal): The list to find from.
        
    Returns:
        (float): The median.
        
    Notes:
        Throws a ZeroDivisionError on an empty list.
    """
    if len(my_list) == 0:
        raise ZeroDivisionError
    
    copied_list = sorted(copy.deepcopy(my_list))
    while len(copied_list) > 2:
        copied_list = copied_list[1:len(copied_list)-1]
        
    if len(copied_list) == 2:
        return (copied_list[0] + copied_list[1]) / 2
    else: # len(copied_list) is 1:
        return copied_list[0]
    
def get_max_indices(my_list):
    """ Gets a list of the maximum values' indices in a list. 
    
    Args:
        my_list (list of ordinal): The list to find the maximums from.
        
    Returns:
        list of int: The indices where max values exist.
    """
    assert len(my_list) > 0
    max_indices = [0]
    curr_max = my_list[0]
    
    for i in range(1, len(my_list)):
        if my_list[i] > curr_max:
            # Reset
            curr_max = my_list[i]
            max_indices = [i]
        elif my_list[i] == curr_max:
            # Add the index
            max_indices.append(i)
            
    return max_indices
        
def get_index(my_list, item):
    """ Gets the index of the item in the list. Returns -1 if not found.
    
    Args:
        my_list (list of obj): The list to search through. Unsorted.
        item: The item to find.
        
    Returns:
        int: The index of the item, or -1 if unfound.
    """
    return my_list.index(item)
    
def convert_to_numeric(table, table_dimensions=2):
    """Try to convert each value in the table to a numeric type (float).
    
    Args:
        table (list of objs): The table to convert. nD List.
        table_dimensions(int): The dimensions of the input table. Assumes 2-D

    Notes:
        Leave values as is that cannot be converted to numeric.
        Recursive if greater than one.
        Returns None if table_dim <= 0
    """
    if table_dimensions <= 0:
        return None
    
    new_table = []
    if table_dimensions == 1: # Exit step
        for item in table:
            if is_float(item):
                item = float(item)
                
            item_copy = copy.deepcopy(item)
            new_table.append(item_copy)
    
    else: # Recursive step
        for row in table:
            new_table.append(convert_to_numeric(row, table_dimensions=table_dimensions-1))
    
    return new_table

def convert_to_integer(table, table_dimensions=2):
    """ Tries to convert every value into an integer where possible
    
    Args:
        table (list of objs): The table to convert. nD List.
        table_dimensions(int): The dimensions of the input table. Assumes 2-D
     """
    if table_dimensions <= 0:
        return None
    
    new_table = []
    if table_dimensions == 1: # Base case
        for item in table:
            if is_int(item):
                item = int(item)
                
            new_table.append(copy.deepcopy(item))
            
    else: # Recursive step
        for row in table:
            new_table.append(convert_to_integer(row, table_dimensions=table_dimensions-1))
            
    return new_table
    
def convert_to_lexical(table, table_dimensions=2):
    """Converts each value in the table to a lexical type (str).
    
    Args:
        table (list of lists): The table to convert. 2D List.
        table_dimensions(int): The dimensions of the input table. Assumes 2-D
    """
    if table_dimensions <= 0:
        return None
    
    new_table = []
    if table_dimensions == 1: # Exit step
        for item in table:
            item = str(item)
            item_copy = copy.deepcopy(item)
            new_table.append(item_copy)
    
    else: # Recursive step
        for row in table:
            new_table.append(convert_to_lexical(row, table_dimensions=table_dimensions-1))
    
    return new_table

def convert_percent_to_numeric(table, table_dimensions=2):
    """ Converts each percentage value in the table to a numeric type (float)
    
    Args:
        table (list of lists): The table to convert. 2D List.
        table_dimensions(int): The dimensions of the input table. Assumes 2-D
    """
    if table_dimensions <= 0:
        return None
    
    new_table = []
    if table_dimensions == 1: # Exit step
        for item in table:
            cop = copy.deepcopy(item)
            cop = str(cop)
            if len(cop) > 1 and cop[-1] == '%' and is_float(cop[:-1]):
                cop = cop[:-1]
                cop = float(cop)
                new_table.append(cop)
                
    else: # Recursive step
        for row in table:
            new_table.append(convert_percent_to_numeric(row, table_dimensions=table_dimensions-1))
            
    return new_table

def convert_rational_to_float(rational):
    """ Convert a rational number in the form of a 2-tuple to a float
    
    Args:
        rational (2-sized list of int): The number to convert
        
    Returns:
        float: The conversion."""
    assert len(rational) == 2
    
    return rational[0] / rational[1]

def load_from_file(filename):
    """Load column names and data from a CSV file.

    Args:
        filename(str): relative path for the CSV file to open and load the contents of.

    Returns:
        list of lists, list of strings: The table and header that was loaded
    
    Notes:
        Use the csv module.
        First row of CSV file is assumed to be the header.
        Calls convert_to_numeric() after load
    """
    table = []
    header = []
    
    # 1. Open the file
    with open(filename, 'r') as f:
        # 2. Parses everything
        filereader = csv.reader(f)
        for row in filereader:
            table.append(row)
    
    # 3. Separates the header
    header = table[0]
    table = table[1:len(table)]
    
    # 4. Converts to numeric
    convert_to_numeric(table)
    
    # 5. Returns the info gained
    return header, table
    
def save_to_file(header, table, filename):
    """Save column names and data to a CSV file.

    Args:
        header(list of str): The column names, and the first row of the csv file
        table(list of lists): The data to be saved to a file
        filename(str): relative path for the CSV file to save the contents to.

    Notes:
        Use the csv module.
    """
    # 1. Makes a copy of the data to convert to strings
    table_copy = copy.deepcopy(table)
    
    # 2. Converts to strings
    convert_to_lexical(table_copy)
    
    # 3. Writes to a csv file, filename
    with open(filename, 'w') as f:
        filewriter = csv.writer(f, delimiter=',')
        filewriter.writerow(header)
        for row in table:
            filewriter.writerow(row)
            
def get_all_unique_values(my_list):
    """Gets all unique values from a list.
    
    Args:
        my_list(list of obj): The list to grab from.
        
    Returns:
        list of obj: The list of unique objects in the list.
    """
    unique_vals = []
    for item in my_list:
        if item not in unique_vals:
            unique_vals.append(item)
            
    unique_vals.sort()
    
    return unique_vals

def group_dict(table, header, attribute, uniques=None, include_indices=False):
    """ Makes a dictionary of new tables that match a certain attribute.
    
    Args:
        table (list of lists): The data to scan through
        header (list of str): The column names
        attribute (str): The column to scan through
        uniques (list of obj): The list of unique values from the list to grab
        include_indices (bool): Whether to also include the indices as a return
        
    Returns:
        dictionary of list of lists: A dictionary mapping attribute strs to tables
    """
    index_folds = []
    
    index = attribute
    if not isinstance(attribute, int):
        index = header.index(attribute)
        
    # If uniques are not passed in, we get it ourselves
    if uniques is None:
        table = transpose(table)
        uniques = get_all_unique_values(table[index])
        table = transpose(table)
        
    if include_indices:
        index_folds = [[] for _ in range(len(uniques))]
        
    d = {}
    for u in uniques:
        d[u] = []
        
    for r in range(len(table)):
        for i in range(len(uniques)):
            if table[r][index] == uniques[i]:
                d[uniques[i]].append(copy.deepcopy(table[r][:index] + table[r][index+1:]))
                if include_indices:
                    # Gets rid of the att_column
                    index_folds[i].append(r[:index] + r[index+1:])
                    
    if include_indices:
        return d, index_folds
    else:
        return d

def group(table, header, attribute, uniques=None, include_indices=False):
    """ Makes a list of new tables that match a certain attribute.
    
    Args:
        table (list of lists): The data to scan through
        header (list of str): The column names
        attribute (str): The column to scan through
        uniques (list of obj): The list of unique values from the list to grab
        
    Returns:
        list of list of lists: A series of tables, grouped by the attribute
    """
    index_folds = []
    
    index = attribute
    if not isinstance(attribute, int):
        index = header.index(attribute)
    
    # If uniques are not passed in, we get it ourselves
    if uniques is None:
        table = transpose(table)
        uniques = get_all_unique_values(table[index])
        table = transpose(table)
        
    if include_indices:
        index_folds = [[] for _ in range(len(uniques))]
    
    supertable = [[] for _ in range(len(uniques))]
    for r in range(len(table)):
        for i in range(len(uniques)):
            if table[r][index] == uniques[i]:
                supertable[i].append(copy.deepcopy(table[r]))
                if include_indices:
                    index_folds[i].append(r)
    
    if include_indices:
        return supertable, index_folds
    else:
        return supertable

def count_in_list(my_list, count_for=1):
    """ Gets the count of a certain value in a list.
    
    Args:
        my_list (list of obj): The list to search through.
        count_for (obj): The object to count.
        
    Returns:
        int: The number of that object
    """
    count = 0
    for item in my_list:
        if item == count_for:
            count += 1
            
    return count

def transpose(my_table):
    """ Transposes a given 2D table.
    
    Args:
        my_table (list of lists): 2D Table.
        
    Returns:
        list of lists: The table transposed.
        
    Notes:
        Requires that all the sublists are the same size.
    """
    if len(my_table[0]) == 0:
        raise ValueError
    
    new_table = []
    for i in range(len(my_table[0])):
        new_row = []
        for j in range(len(my_table)):
            new_row.append(copy.deepcopy(my_table[j][i]))
        new_table.append(copy.deepcopy(new_row))
        
    return new_table

def eliminate_incomplete_instances(parallel_lists):
    """ Gets rid of indices in each list that's missing.
    
    Args:
        parallel_lists (list of lists): List with parallel lists.
        
    Returns:
        (list of lists): List of unmissing parallel lists.
    """
    complete = []
    # We now transposed it
    transposed = transpose(parallel_lists)
    
    for row in transposed:
        add_to = True
        for item in row:
            if item == "NA" or item == "N/A" or item == "":
                add_to = False
                break
        
        if add_to:
            complete.append(copy.deepcopy(row))
            
    # Gotta transpose it again
    return transpose(complete)

def merge(mylist, left, mid, right, sort_on=None, parallel_lists=False, mode="asc_left"):
    """ HELPER for merge_sort function """

    n1 = mid - left + 1
    n2 = right - mid
    
    A1 = [copy.deepcopy(mylist[left + i]) for i in range(n1)]
    A2 = [copy.deepcopy(mylist[mid + 1 + i]) for i in range(n2)]

    i1, i2 = 0, 0
    if not parallel_lists:
        while i1 < n1 and i2 < n2:
            # Minor operator changes
            if mode == "asc_left":
                if A1[i1] <= A2[i2]:
                    mylist[left + i1 + i2] = A1[i1]
                    i1 += 1
                else:
                    mylist[left + i1 + i2] = A2[i2]
                    i2 += 1

            elif mode == "desc_left":
                if A1[i1] >= A2[i2]:
                    mylist[left + i1 + i2] = A1[i1]
                    i1 += 1
                else:
                    mylist[left + i1 + i2] = A2[i2]
                    i2 += 1

            elif mode == "asc_right":
                if A1[i1] < A2[i2]:
                    mylist[left + i1 + i2] = A1[i1]
                    i1 += 1
                else:
                    mylist[left + i1 + i2] = A2[i2]
                    i2 += 1

            elif mode == "desc_right":
                if A1[i1] > A2[i2]:
                    mylist[left + i1 + i2] = A1[i1]
                    i1 += 1
                else:
                    mylist[left + i1 + i2] = A2[i2]
                    i2 += 1

            else:
                raise ValueError("Invalid mode given.")

        # Catch the leftovers
        if i1 == n1:
            while i2 < n2:
                mylist[left + i1 + i2] = A2[i2]
                i2 += 1
        elif i2 == n2:
            while i1 < n1:
                mylist[left + i1 + i2] = A1[i1]
                i1 += 1

        return
    
    else:
        while i1 < n1 and i2 < n2:
            # Minor operator changes
            if mode == "asc_left":
                if A1[i1][sort_on] <= A2[i2][sort_on]:
                    mylist[left + i1 + i2] = A1[i1]
                    i1 += 1
                else:
                    mylist[left + i1 + i2] = A2[i2]
                    i2 += 1

            elif mode == "desc_left":
                if A1[i1][sort_on] >= A2[i2][sort_on]:
                    mylist[left + i1 + i2] = A1[i1]
                    i1 += 1
                else:
                    mylist[left + i1 + i2] = A2[i2]
                    i2 += 1

            elif mode == "asc_right":
                if A1[i1][sort_on] < A2[i2][sort_on]:
                    mylist[left + i1 + i2] = A1[i1]
                    i1 += 1
                else:
                    mylist[left + i1 + i2] = A2[i2]
                    i2 += 1

            elif mode == "desc_right":
                if A1[i1][sort_on] > A2[i2][sort_on]:
                    mylist[left + i1 + i2] = A1[i1]
                    i1 += 1
                else:
                    mylist[left + i1 + i2] = A2[i2]
                    i2 += 1

            else:
                raise ValueError("Invalid mode given.")

        # Catch the leftovers
        if i1 == n1:
            while i2 < n2:
                mylist[left + i1 + i2] = A2[i2]
                i2 += 1
        elif i2 == n2:
            while i1 < n1:
                mylist[left + i1 + i2] = A1[i1]
                i1 += 1

        return

    # End of merge def

def merge_sort(mylist, sort_on=None, parallel_lists=False, mode="asc_left", start=None, end=None):
    """ Performs a simple in-place merge sort upon mylist.
    
    Args:
        mylist (list of obj or list of list of obj): 1- or 2-D list of ordinal objects to sort
        sort_on (int or None): The index to sort on, if 2D
        parallel_lists (bool): Whether or not mylist is 2D
        mode (str): The mode to sort with. Normally ascending.
        start (int): The index to start on. Inclusive.
        end (int): The index to end with. Inclusive.
        
    Notes:
        Assumes the lists are of the same size
    """    
    if parallel_lists and sort_on is None:
        raise ValueError("Must include an index to sort on")
    
    if start is None:
        start = 0;
        
    if end is None:
        end = len(mylist) - 1
        
    # Our recursive exit step
    if start >= end:
        return
    
    mid = (start + end) // 2
    merge_sort(mylist, sort_on=sort_on, parallel_lists=parallel_lists, mode=mode, start=start, end=mid)
    merge_sort(mylist, sort_on=sort_on, parallel_lists=parallel_lists, mode=mode, start=mid+1, end=end)
    merge(mylist, start, mid, end, sort_on=sort_on, parallel_lists=parallel_lists, mode=mode)
        
def parallel_shuffle(mylists):
    """ Shuffles parallel lists together
    
    Args:
        mylists (list of list of obj): The list of parallel lists to shuffle.
            Note that the format is [l1, l2, ..., ln]
    
    Returns:
        list of list of obj: The list of shuffled parallel lists.
        
    Notes:
        Random should be seeded beforehand.
    """
    num = len(mylists)
    length = len(mylists[0])
    length_array = list(range(length))
    random.shuffle(length_array)
    
    shuffled_lists = []
    for i in range(num):
        shuffled_lists.append([])
        
    for order in length_array:
        for i in range(num):
            shuffled_lists[i].append(mylists[i][order])
    
    return shuffled_lists

def scale(mylist, zero_vals=None, one_vals=None):
    """ Scales a 2D list by attribute from 0 to 1
    
    Args:
        mylist (list of list of obj): The lists to scale
        zero_vals (list of numbers): The override of the minimum values to scale to zero with.
        one_vals (list of numbers): See zero_vals. Both or neither must be given by the user.
        
    Returns:
        mins (list of numbers): The minimum found for each attribute
        maxs (list of numbers): The maximum found for each attribute
    """
    maxs = []
    mins = []
    if zero_vals is None and one_vals is None:
        for i in range(len(mylist[0])):
            maxs.append(mylist[0][i])
            mins.append(mylist[0][i])

        # Get the mins and maxes
        for i in range(len(mylist)):
            for j in range(len(mylist[i])):
                if mylist[i][j] > maxs[j]:
                    maxs[j] = mylist[i][j]
                elif mylist[i][j] < mins[j]:
                    mins[j] = mylist[i][j]
    elif zero_vals is None or one_vals is None:
        raise ValueError("Cannot have one optional arg but not the other.")
    else:
        maxs = one_vals
        mins = zero_vals
    
    # Scale appropriately
    for i in range(len(mylist)):
        for j in range(len(mylist[i])):
            mylist[i][j] = (mylist[i][j] - mins[j]) / (maxs[j] - mins[j])
            
    return mins, maxs

def evaluate_classifier(X_train, y_train, X_test, y_test, classifier):
    """ Computes the accuracy of a given classifier given it has fit() and predict() functions.
    
    Args:
        X_train (list of list of obj): The X train set
        y_train (list of obj): The y train set
        X_test (list of list of obj): The X test set
        y_test (list of obj): The y test set
        classifier (Classifier object, with fit and predict): The classifier to evaluate
        
    Returns:
        float: The accuracy of the classifier.
    """
    # Have to make sure
    assert len(y_test) > 0
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    numerator = 0
    denominator = 0
    for i in range(len(y_pred)):
        denominator += 1
        if y_pred[i] == y_test[i]:
            numerator += 1
    
    return numerator / denominator

def equivalent(list1, list2):
    """ Checks if two (possibly nested) lists are equivalent. Recursive.
    
    Args:
        list1, list2 (lists): The lists to compare
        
    Returns:
        bool: Whether the lists are equivalent
    """
    
    # Base case
    if not isinstance(list1, list) and not isinstance(list2, list):
        return type(list1) == type(list2) and list1 == list2
    # Exit case
    elif not isinstance(list1, list) or not isinstance(list2, list) or len(list1) != len(list2):
        return False
    # Recursive case
    else:
        for i in range(len(list1)):
            if not equivalent(list1[i], list2[i]):
                return False # Looks redundant, but it isn't. We need to check for each.
    
    # If we got through everything else. Will only happen in list cases.
    return True

def get_cutoff_frequencies(values,cutoffs):
    """ The purpose of this function to calculate the frequencies of values in a column based off of cutoff values.
    Attributes:
        - values(list): the list of values which we want to get values from.
        - cutoffs(list): list of values to have cutoffs
    Returns:
        - counts(list): frequencies of values.
    """
    counts = [0 for _ in range(len(cutoffs)-1)]
    for value in values:
        contains_value = False
        for i in cutoffs[1:]:
            if value < i:
                contains_value= True 
                counts[cutoffs.index(i)-1] += 1
                break
        if not contains_value:
            counts[-1] += 1
    return counts


def compute_equal_width_cutoffs(values, num_bins):
    """ Computes where cutoffs should occur based off a desired number of bins and values within a list.
    Attributes:
        - values(list): column of values which cutoffs will be created based off
        - num_bins(int): number of desired bins
    Returns:
        - cutoffs(list): list of cutoff values
    """
    # first compute the range of the values
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins 
    # bin_width is likely a float
    # if your application allows for ints, use them
    # we will use floats
    # np.arange() is like the built in range() but for floats
    cutoffs = list(np.arange(min(values), max(values), bin_width)) 
    cutoffs.append(max(values))
    return cutoffs

def discretize(data, num_bins, cutoffs=None):
    """ Discretizes the continuous-valued list given with a modular focus (namely, num_bins)
    
    Args:
        data (list of lists of ordinal): The 2d table to discretize (by each attribute)
        num_bins (list of natural num): The number of bins for each attribute to be split into
        cutoffs (list of None or list of ordinal): The override for cutoff points.
        
    Returns:
        list of list of natural num): The discretized data.
    """
    # First, we need to check something...
    assert isinstance(num_bins, list)
    
    # Now we can start. First, we use the cutoffs override appropriately to set data intelligently...
    if not isinstance(cutoffs, list):
        assert cutoffs is None
        # The default; We do this manually
        cutoffs = []
        for c in range(len(num_bins)):
            assert num_bins[c] > 0
            cutoffs.append(copy.deepcopy(compute_equal_width_cutoffs(get_column(data, None, c), num_bins[c])))
    else:
        # The given cutoffs
        # Here, we want to check if at each index of cutoffs:
        # A) If the index is None, we get it
        # B) If the num_bins is fine
        assert len(cutoffs) == len(num_bins)
        for c in range(len(cutoffs)):
            if cutoffs[c] is None:
                cutoffs[c] = copy.deepcopy(compute_equal_width_cutoffs(get_column(data, None, c), num_bins[c]))
            else:
                assert len(cutoffs[c]) == num_bins[c] + 1
                
    # Before we move forward, I'd like to remove the first and last number in cutoffs; Why?
    # We may get values that are above or below our maxes and mins as given. As such we want to try and
    # classify those as respectively the highest and lowest discretization
    for c in range(len(cutoffs)):
        cutoffs[c] = cutoffs[c][1:-1]
                
    # Okay. Now we have appropriate cutoffs. Let's use them to properly discretize our data.
    for row in data:
        for c in range(len(row)):
            # We have our datapoint...
            # We need to find what number it belongs to
            found = False
            for p in range(len(cutoffs[c])):
                if not found and row[c] < cutoffs[c][p]:
                    row[c] = p+1
                    found = True
                    
            if not found:
                row[c] = len(cutoffs[c])+1
                
    return data

def compute_euclidean_distance(v1, v2):
    """ The purpose of this function is to compute the euclidean distance of two given values
    Args:
        - v1(int,float or string): x1 value
        - v2(int,float, or string): x2 value
    Returns:
        - dist(float): distance between two values
    """
    assert len(v1) == len(v2)
    dist = 0
    if isinstance(v1[0],str) and isinstance(v2[0],str):
        for i in range(len(v1)):
            if v1[i] == v2[i]:
                dist += 0
            elif v1[i] != v2[i]:
                dist += 1
    else:
        dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist 

def get_frequencies(column):
    """ The purpose of this function is to get the count for how many times a value appeaars in a given column.
    Attributes:
        - column(list): a list of values in a column which will be checked for values and frequencies.
    Returns:
        - values, counts(tuple of lists): a tuple containing the values within a column and their associated frequencies. 
    """
    values = []
    counts = []
    column.sort()
    for value in column:
        if value not in values:
            values.append(value)
            counts.append(1)
        else:
            counts[-1] += 1
    return values, counts
