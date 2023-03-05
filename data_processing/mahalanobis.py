import numpy as np
import pandas as pd
from scipy.stats import chi2

def mahalanobis_with_model(data_matrix: object, mean_vector: object, cov_matrix: object):
    """Take a numpy 2darray and calculate the Mahalanobis distance for all features"""
    # Get the shape of the data
    [features, observations] = np.shape(data_matrix)
    # Create a matrix of size [feature X observations] where each column is a copy of the mean vector
    mean_matrix = mean_vector
    for i in range(observations-1):
        mean_matrix = np.column_stack((mean_matrix, mean_vector))

    # Take the transpose of the data matrix - the mean matrix
    XC_transpose = np.transpose(data_matrix - mean_matrix)

    # Use matrix multiplication to calculate the product of the previous calculation and the inverse covariance matrix
    first_calculation = np.matmul(XC_transpose, np.linalg.inv(cov_matrix))

    # Use element-wise matrix multiplication to calculate the product of the
    # previous calculation and the XC_transpose matrix
    final_calculation = np.multiply(first_calculation, XC_transpose)
    # Return the sum of the final calculation across all rows of the data
    return np.sum(final_calculation, axis=1)

def mahalanobis(data: object):
    """Take a numpy 2darray and calculate the Mahalanobis distance for all features"""
    # Transpose the matrix so the data is of shape num features X num observations
    data_matrix = np.transpose(data)
    # Calculate the mean vector of size [1 X dim]
    mean_vector = np.mean(data_matrix, axis = 1)
    # Calculate the covariance matrix
    cov_matrix = np.cov(data_matrix)
    
    return mahalanobis_with_model(data_matrix, mean_vector, cov_matrix)

def outlier_removal(df: object, class_features: list, label: str, threshold=None):
    """
    Remove the value with the highest mahalanobis distance sequentially
    or provide a threshold to remove multiple values. A chi-squared 
    distrubution is used to calculate the cutoff because mahalanois
    returns the distances squared. Pass the threshold value as a float
    i.e: 0.95
    df should be of shape num obs X num features
    """
    class_labels = np.array(df[label])
    class_obs = np.array(df.drop([label], axis=1))
    
    mahal = mahalanobis(class_obs)
    outlier_indices = []
    
    # Get indices to remove. Remove one outlier if the threshold is none.
    if threshold == None:
        sorted_mahal = sorted(mahal)
        outlier = sorted_mahal.pop()
        outlier_indices = np.where(mahal == outlier)
    else:
        cutoff = chi2.ppf(threshold, class_obs.shape[1])
        outlier_indices = np.where(mahal > cutoff )
    
    # axis should be the number of observations
    new_class_obs = np.delete(class_obs, outlier_indices, axis=0)
    new_class_labels = np.delete(class_labels, outlier_indices, axis=0)
        
    new_df = pd.DataFrame(new_class_obs, columns=class_features)
    new_df[label] = new_class_labels
    return new_df
        
        