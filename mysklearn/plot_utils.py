import matplotlib.pyplot as plt
import importlib
import math
import numpy as np

def create_histogram(name,data):
    """ The purpose of this function is to create a histogram
    Attributes:
        - name(string): name of chart
        - data(list): list of values to create histogram based off of
    Returns:
        - Nothing
    """
    # data is a 1D list of data values
    plt.figure()
    plt.hist(data, bins=10) # default is 10
    plt.title("{} Histrogram".format(name))
    plt.xlabel("{} Values".format(name))
    plt.ylabel("Counts")
    plt.show()


def find_covarient(x,y):
    """ The purpose of this function is to calculate the covarient of a scatter plot.
    Attributes:
        - x(list): a list of a column which is graphed. The x axis values.
        - y(list): a list of a column which is graphed. The y axis values.
    Returns:
        - cov(float): the covalent value of a graph
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    cov = sum([(x[i]-mean_x)*(y[i]-mean_y) for i in range(len(x))]) / len(x)
    return round(cov,2)

def find_correlation(x,y,cov):
    """ The purpose of this function is to calculate the correlation value of a graph.
    Attributes:
        - x(float): std deviation value of x value column
        - y(float): std deviation value of y value column
        - cov(float): covalent value of graph
    Returns:
        - correlation(float): the correlation value of a graph.
    """
    return round((cov / (x * y)),2)


def compute_slope_intercept(x, y):
    """ Computes the slope intercept of a line
    Attributes:
        - x(list): x value column
        - y(list): y value column
    Returns:
        - m,b(tuple): m is slope and b is intercept
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => b = y - mx
    b = mean_y - m * mean_x
    return m, b 

def find_standard_deviation(column):
    """ The purpose of this function is to calculate the standard deviation of a given column.
    Attributes:
        - column(list): a list of the column to have the standard deviation calculated for.
    Returns:
        - std_dev(float): the std_dev value of a graph.
    """
    mean = sum(column) / len(column)
    squared_mean_deviations = [(x - mean) ** 2 for x in column]
    variance = sum(squared_mean_deviations) / len(squared_mean_deviations)
    std_dev = math.sqrt(variance)
    return std_dev


