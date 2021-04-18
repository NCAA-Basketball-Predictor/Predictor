import matplotlib.pyplot as plt
import importlib
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

