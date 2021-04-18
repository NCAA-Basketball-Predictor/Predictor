"""
Programmer: Gina Sprint and Brandon Clark
Class: CPSC 322-02, Spring 2021
Programming Assignment #6
03/25/2021
I did not attempt the bonus
Description: This program contains the functions neccessary to do data preparation.
"""
import copy
import csv 
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names) # TODO: fix this

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            tuple of int: rows, cols in the table
        Notes:
            Raise ValueError on invalid col_identifier
        """ 
        col = []
        if isinstance(col_identifier,str):
            col_index = self.column_names.index(col_identifier)
        elif isinstance(col_identifier,int):
            col_index = col_identifier
        else:
            raise ValueError("Invalid input")
        for row in self.data:
            if include_missing_values == True:
                col.append(row[col_index])
            elif include_missing_values == False:
                if(row[col_index] != "NA"):
                    col.append(row[col_index])
        return col # TODO: fix this
    

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    numeric_value = float(self.data[i][j])
                    self.data[i][j] = numeric_value
                except ValueError:
                    continue

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.
        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        index = 0
        while index < len(self.data):
            for row in rows_to_drop:
                if self.data[index] == row:
                    del self.data[index]
            else:
                index += 1
    
    def drop_column(self,col_identifier):
        '''Drops a column from a list
        Args:
            - col_identifier(int or string): the identifier of the column from a list
        Returns:
            - new_table(list of lists): the new table without the undesired column
        '''
        if isinstance(col_identifier,str):
            col_index = self.column_names.index(col_identifier)
        elif isinstance(col_identifier,int):
            col_index = col_identifier
        else:
            raise ValueError("Invalid input")
        new_table = copy.deepcopy(self.data)
        for i,row in enumerate(self.data):
            new_table[i].pop(col_index)
        return new_table

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        rows = []
    # 1. open
        infile = open(filename, "r")
    # 2. process (read/write)
        csv_reader = csv.reader(infile)
        for row in csv_reader:
            rows.append(row)
        self.column_names = rows[0]
        del rows[0]
        self.data = copy.deepcopy(rows)
        self.convert_to_numeric()
        
        # 3. close
        infile.close()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(self.column_names)
        csv_writer.writerows(self.data)
        outfile.close

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicates = []
        duplicate = True
        for row_1 in range(len(self.data)):
            for row_2 in range(len(self.data)):
                if row_1 != row_2:
                    duplicate = True
                    for key in key_column_names:
                        index = self.column_names.index(key)
                        if self.data[row_1][index] != self.data[row_2][index]:
                            duplicate = False
                    if duplicate:
                        if self.data[row_1] not in duplicates and self.data[row_2] not in duplicates:
                            duplicates.append(self.data[row_2])
        return duplicates # TODO: fix this

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        index = 0
        delete_row = False
        while index < len(self.data):
            delete_row = False
            for i in range(len(self.data[index])):
                if(self.data[index][i] == "NA"):
                    delete_row = True

            if delete_row:    
                del self.data[index]
            else:
                index += 1

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        sum = 0
        count = 0
        for i in range(len(self.data)):
            index = self.column_names.index(col_name)
            if self.data[i][index] != "NA":
                sum += self.data[i][index]
                count += 1
        average = sum / count
        for i in range(len(self.data)):
            index = self.column_names.index(col_name)
            if self.data[i][index] == "NA":
                self.data[i][index] = average
    
    def median(self,list):
        """Finds the median of a row in dataset
        Args:
            - list(a list): list to find the median in
        """
        n = len(list)
        s = sorted(list)
        return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None
    
    def average(self,list):
        """ Computes the average of a column in the dataset 
        Args:
            - list(a list): list to find the average of
        """
        sum = 0
        count = 0
        for i in list:
            sum += i
            count += 1
        return sum / count 

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed.
        """
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        data = []
        for column in col_names:
            data.append([column,"NA","NA","NA","NA","NA"])
        if len(self.data) != 0:
            for row in data:
                col = self.get_column(row[0], False)
                row[1] = min(col)
                row[2] = max(col)
                row[3] = (row[1] + row[2]) / 2
                row[4] = self.average(col)
                row[5] = self.median(col)
        else:
            data = []
        return MyPyTable(header, data) # TODO: fix this
    
  

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table = []
        header = copy.deepcopy(self.column_names)
        for column in other_table.column_names:
            if column not in header:
                header.append(column)
        
        in_both = True
        for row_1 in range(len(self.data)):
            for row_2 in range(len(other_table.data)):
                in_both = True
                for key in key_column_names:
                    index_1 = self.column_names.index(key)
                    index_2 = other_table.column_names.index(key)
                    if self.data[row_1][index_1] != other_table.data[row_2][index_2]:
                        in_both = False
                if in_both:
                    added_row = copy.deepcopy(self.data[row_1])
                    for column in other_table.column_names:
                        if column not in self.column_names:
                            adding_index = other_table.column_names.index(column)
                            added_row.append(other_table.data[row_2][adding_index])
                    joined_table.append(added_row)
        return MyPyTable(header,joined_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        header = copy.deepcopy(self.column_names)
        for column in other_table.column_names:
            if column not in header:
                header.append(column)
        
        joined_table = []
        in_both = True
        for row in range(len(self.data)):
            added_row = ["NA" for x in range(len(header))]
            for column in header:
                column_index = header.index(column)
                if column in self.column_names:
    
                    added_row[column_index] = self.data[row][self.column_names.index(column)]
            joined_table.append(added_row)
        is_updated = False
        for row_2 in range(len(other_table.data)):
            is_updated = False
            for row_3 in range(len(joined_table)):
                in_both = True
                for key in key_column_names:
                    index_1 = header.index(key)
                    index_2 = other_table.column_names.index(key)
                    if joined_table[row_3][index_1] != other_table.data[row_2][index_2]:
                        in_both = False
                if in_both:
                    is_updated = True 
                    for column in other_table.column_names:
                        adding_index = other_table.column_names.index(column)
                        joined_table[row_3][header.index(column)] = other_table.data[row_2][adding_index]
            if is_updated == False:
                added_row_2 = ["NA" for x in range(len(header))]
                for column_2 in header:
                    column_index = header.index(column_2)
                    if column_2 in other_table.column_names:
                        added_row_2[column_index] = other_table.data[row_2][other_table.column_names.index(column_2)]
                joined_table.append(added_row_2)
        outer_join_table = MyPyTable(header, joined_table)
        return outer_join_table # TODO: fix this