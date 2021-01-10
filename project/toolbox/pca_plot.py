import matplotlib.pyplot as plt
import numpy as np

def get_variance_percentage(pca):
    """
    Retruns the pca variance percentage

    Parameters
    ----------
    pca : PCA
        principle components
        """
    pca_variance_ratio = pca.explained_variance_ratio_.cumsum()
    pca_variance_percentage = pca_variance_ratio * 100
    return pca_variance_percentage

def get_number_of_attributes(pca_variance_percentage):
    """
    Retruns the number of pca components
    Parameters
    ----------
    pca_variance_percentage : numpy array
        calculated variance of the principle components
        """
    #Determine number of attributes needed for an explained variance of 90%
    pca_variance_cropped = [i for i in pca_variance_percentage if i < 90]
    no_attributes = len(pca_variance_cropped)
    return no_attributes

def get_pca_data_sets(pca_complete, no_attributes):
    #Crop pca_data
    pca_data = np.delete(pca_complete, slice(no_attributes, len(pca_complete)) , 1)
    
    #Split data back to original train and test split
    pca_train = pca_data[:38]
    pca_test = pca_data[38:]
    
    return pca_data, pca_train, pca_test

def plot_pca_variance(pca_variance_percentage, no_attributes):
    """
    Plots pca variance
    Parameters
    ----------
    pca_variance_percentage : numpy array
        calculated variance of the principle components
    no_attributes : int
        number of principle components that explain the variance
        """
    plt.bar(range(1,pca_variance_percentage.size+1), pca_variance_percentage)
    plt.title("Explained variance summed per attribute")
    plt.xlabel("Attribute number")
    plt.ylabel("Explained variance percentage")
    plt.show()
    print('''Figure: A plot of the explained variance. The variance is summed for all attributes up to and including
    the current attribute number,for examplethe tenth attribute shows the sum of attributes 1 to 10.''')
    print("There are {} attributes that together explain 90% of the variance." .format(no_attributes))