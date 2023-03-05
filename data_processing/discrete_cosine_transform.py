import numpy as np
from scipy.fftpack import dct

def get_2d_dct_both_axes(data):
    """
    Get 2d Discrete Cosine Transform
    data: a numpy array of images
    """
    dct_array = []
    # Get the 2d discrete cosine transform across both axes
    for img in data:
        dct_array.append(dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho'))

    return np.array(dct_array)

# Element-wise multiplication with the masks to obtain the directional coefficients of the dcts
def get_directional_coefficients(dct_transforms, vertical_mask, diagonal_mask, horizontal_mask):
    vertical_coeffs = np.array([np.multiply(image, vertical_mask) for image in dct_transforms])
    horizontal_coeffs = np.array([np.multiply(image, horizontal_mask) for image in dct_transforms])
    diagonal_coeffs = np.array([np.multiply(image, diagonal_mask) for image in dct_transforms])
    return [vertical_coeffs, diagonal_coeffs, horizontal_coeffs]

def flatten_and_get_non_zeros(data: object):
    """Flatten the data passed in and get non-zero indices"""
    flattened_data = []
    for img in data:
        flattened_data.append(np.ndarray.flatten(img[img != 0]))
            
    # transpose the data to match the matlab exmaple   
    return np.array(flattened_data).T

def explained_variance_ratios(eigvals):
    explained_variance_ratios = []
    total = sum(eigvals)
    for i in range(len(eigvals)):
        explained_variance_ratios.append(eigvals[i]/sum(eigvals))
    return explained_variance_ratios

def get_directional_cov_matrices(flattened_vert_coeffs, flattened_horiz_coeffs, flattened_diag_coeffs):
    """
    Use the flattened data to get directional covariance matrices
    """
    cov_v = np.cov(flattened_vert_coeffs.T - np.mean(flattened_vert_coeffs.T, axis=0), rowvar=False)
    cov_h = np.cov(flattened_horiz_coeffs.T - np.mean(flattened_horiz_coeffs.T, axis=0),rowvar=False)
    cov_d = np.cov(flattened_diag_coeffs.T - np.mean(flattened_diag_coeffs.T, axis=0),rowvar=False)
    return [cov_v, cov_h, cov_d]