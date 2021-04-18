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


def draw_scatter_plot(x,y,x_name,y_name):
    plt.figure()
    m,b = compute_slope_intercept(x,y)
    plt.scatter(x,y)
    covarient = find_covarient(x,y)
    x_standard_deviation = find_standard_deviation(x)
    y_standard_deviation = find_standard_deviation(y)
    correlation = find_correlation(x_standard_deviation,y_standard_deviation,covarient)
    plt.annotate("corr: {}, cov: {}".format(correlation,covarient), xy=(0.17, 0.93), xycoords="axes fraction", horizontalalignment="center", color="red",bbox=dict(boxstyle="round", fc="1", color="r"))
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r",lw=5)
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.title(x_name+" V.S. "+y_name)
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
    # optionally: might want to round
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs

