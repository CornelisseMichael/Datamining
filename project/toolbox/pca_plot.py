import matplotlib.pyplot as plt
def get_variance_percentage(pca):
    pca_variance_ratio = pca.explained_variance_ratio_.cumsum()
    pca_variance_percentage = pca_variance_ratio * 100
    return pca_variance_percentage

def get_number_of_attributes(pca_variance_percentage):
    #Determine number of attributes needed for an explained variance of 90%
    pca_variance_cropped = [i for i in pca_variance_percentage if i < 90]
    no_attributes = len(pca_variance_cropped)
    return no_attributes

def plot_pca_variance(pca_variance_percentage, no_attributes):
    plt.bar(range(1,pca_variance_percentage.size+1), pca_variance_percentage)
    plt.title("Explained variance summed per attribute")
    plt.xlabel("Attribute number")
    plt.ylabel("Explained variance percentage")
    plt.show()
    print('''Figure: A plot of the explained variance. The variance is summed for all attributes up to and including
    the current attribute number,for examplethe tenth attribute shows the sum of attributes 1 to 10.''')
    print("There are {} attributes that together explain 90% of the variance." .format(no_attributes))