"""
A python file to store common helper functions used throughout the problems
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For confidence ellipses
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confusion_matrix(test_labels, predictions):
    """Display confusion matrix"""
    test_labels = pd.Series(test_labels, name='Actual')
    y_pred = pd.Series(predictions, name='Predicted')
    df_confusion = pd.crosstab(test_labels, y_pred)
    print("Confusion Matrix:\n")
    print(f"{df_confusion}\n")
    
def print_accuracy(labels, classifications):
    correct_pred = 0
    for i, val in enumerate(classifications):
        if classifications[i] == labels[i]:
            correct_pred += 1
            
    accuracy = (correct_pred/len(labels))*100;
    print(f"Accuracy: {round(accuracy, 2)}%")
    
def get_accuracy(labels, classifications):
    correct_pred = 0
    for i, val in enumerate(classifications):
        if classifications[i] == labels[i]:
            correct_pred += 1
            
    accuracy = (correct_pred/len(labels))*100;
    return round(accuracy, 2)
    
def shuffle_data(df: object) -> object:
    """Shuffle a dataframe"""
    return df.sample(frac=1).reset_index(drop=True)

def split_by_class(df: object, classes: list, label: str, num_examples=None) -> dict:
    """Split data by class"""
    split_by_class = {}
    for class_type in classes:
        if num_examples != None:
            split_by_class[class_type] = df.loc[df[label] == class_type].iloc[:num_examples, :]
        else:
            split_by_class[class_type] = df.loc[df[label] == class_type]
    
    return split_by_class
               
def reformat_df_by_class(class_data: list):
    """Add all classes back to the dataframe and randomize"""
    df = pd.concat(class_data)
    return df.sample(frac=1).reset_index(drop=True)

# re-encode labels as categorical in dataframe
def df_labels_to_numerical(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df['class_int'] = pd.Categorical(df[label]).codes
    return df

def get_class_combinations(labels: list)->list:
    """Get all combinations of classes in a dataset"""
    combinations = []
    # Collect all combinations of numbers
    for val_1 in labels:
        for val_2 in labels:
            if val_1 != val_2 and [val_2, val_1] not in combinations:
                combinations.append([val_1, val_2])
    return combinations

def get_feature_combinations(features: list) ->list:
    """Get all combinations of features"""
    combinations = []
    # Collect all combinations of features
    for feature_1 in features:
        for feature_2 in features:
            if feature_1 != feature_2 and [feature_2, feature_1] not in combinations:
                combinations.append([feature_1, feature_2])
    return combinations

def plot_class_distributions_by_feature(df:object, feature:str, classes: list, colors:dict):
    plt.rcParams["figure.figsize"] = [5, 5]
    for class_type in classes:
        class_obs = df[df["species"]==class_type]
        feature_df = class_obs.loc[:, [feature]]
        plt.hist(np.array(feature_df), color=colors[class_type])
        plt.title(f"{feature} by class")
        plt.legend(classes)

def split_train_test(data: object, labels: object, test_percent: float):
    """Split data into training and testing sets"""
    len_data = len(data)
    split_val = int(len_data - len(data) * test_percent)
    train_data = data.iloc[:split_val, :]
    test_data = data.iloc[split_val:, :]
    train_labels = labels.iloc[:split_val]
    test_labels = labels.iloc[split_val:]
    
    return train_data, test_data, train_labels, test_labels

def split_labels(data: object, label_name: str) -> list:
    """
    Split labels and data
    """
    labels = data[label_name]
    new_data = data.drop([label_name], axis=1)
    return labels, new_data

"""
Plot with confidence ellipses
"""
# Confidence Ellipse code from Python code discussion area: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html 
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def two_feature_ce_visualization(df: object, feature_1: str, feature_2: str, classes, ax):
    """Visualize two features, grouped by class, using the confidence ellipse function above"""
    colors = {"setosa": "red", "versicolor": "blue", "virginica": "orange"}
    
    for class_type in classes:
        class_df = df[df["species"] == class_type]
        x, y = class_df.loc[:, feature_1], class_df.loc[:, feature_2]
        ax.scatter(x, y, s=0.5, color= colors[class_type])
        confidence_ellipse(x, y, ax, label=class_type, edgecolor=colors[class_type])
    ax.set_title(f"{feature_1} vs. {feature_2}")
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.legend()

def plot_feature_combinations(data: object, feature_permutations: list, labels: list):
    """Create subplots to plot all feature combinations"""
    rows, cols = 3, 2
    fig, ax = plt.subplots(rows, cols, figsize=(10, 10))
    
    two_feature_ce_visualization(data, feature_permutations[0][0], feature_permutations[0][1], labels, ax[0, 0])
    two_feature_ce_visualization(data, feature_permutations[1][0], feature_permutations[1][1], labels, ax[0, 1])
    two_feature_ce_visualization(data, feature_permutations[2][0], feature_permutations[2][1], labels, ax[1, 0])
    two_feature_ce_visualization(data, feature_permutations[3][0], feature_permutations[3][1], labels, ax[1, 1])
    two_feature_ce_visualization(data, feature_permutations[4][0], feature_permutations[4][1], labels, ax[2, 0])
    two_feature_ce_visualization(data, feature_permutations[5][0], feature_permutations[5][1], labels, ax[2, 1])
    plt.tight_layout()                     
    plt.show()
    
def plot_clusters(data, clusters, colors):
    if len(clusters) > len(colors):
        print("The colors list should have a color for each cluster")
        return
    
    for i in range(len(clusters)):
        # get the true value for the cluster values from dataset
        values = data[data.index.isin(clusters[i])]
        plt.plot(values.iloc[:, 1], values.iloc[:, 2], colors[i])

def stack_columns(array_1: object, array_2: object):
    """Stack two numpy arrays column-wise"""
    # Check if appending to an empty array
    if len(array_1) == 0:
        new_data = array_2
    else:
        new_data = np.column_stack((array_1, array_2))
    
    return new_data

def get_class_indices(data, labels):
    """
    Get the indices for all classes in a numpy array
    containing class labels
    """
    label_indices = {}
    for label in labels:
        label_indices[label] = np.where(data == label)[0]
    return label_indices

"""
Helpers for image data
"""
def reshape_image(image, dimesions):
    """
    reshape image, dimensions should be in the form (m, n)
    """
    return np.reshape(image, dimesions)

def plot_sample_images(data: np.array, indices: list):
    """Function to plot the sample images at the given indices"""

    # Set the image size
    plt.rcParams["figure.figsize"] = [10, 5]

    # Set the starting index for the 2 X 5 plot of images
    starting_fig_index = 0

    # Add the images to the subplot and display
    for index in indices:
        starting_fig_index += 1
        plt.subplot(2, 5, starting_fig_index)
        plt.imshow(data[index][:], cmap='gray')
        plt.title(f"Index: {index}")

"""
Helpers for the built-in PCA algorithm in sklearn
"""
def plot_two_principal_components(two_component_pca, labels, classes, colors):
    # Visualization code from https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    principal_df = pd.DataFrame(data = two_component_pca
             , columns = ['principal component 1', 'principal component 2'])
    final_df = pd.concat([principal_df, labels], axis = 1)

    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    for target, color in zip(classes,colors):
        indices_to_keep = final_df['species'] == target
        ax.scatter(final_df.loc[indices_to_keep, 'principal component 1']
               , final_df.loc[indices_to_keep, 'principal component 2']
               , c = color
               , s = 50)
    ax.legend(classes)
    ax.grid()
    
def pca_variance_and_scree_analysis(data: object, components: int):
    """Analyze the data with principal component analysis"""
    pca = PCA(n_components=4)
    pca_model = pca.fit(data)
    
    print(f"The first component accounts for {round(pca.explained_variance_ratio_[0]*100, 2)}% of the variance")
    print(f"The second component accounts for {round(pca.explained_variance_ratio_[1]*100, 2)}% of the variance")
    print(f"The third component accounts for {round(pca.explained_variance_ratio_[2] * 100, 2)}% of the variance")
    print(f"The fourth component accounts for {round(pca.explained_variance_ratio_[3]*100, 2)}% of the variance")
    
    PC_values = np.arange(pca.n_components_) + 1
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()
    
    return pca.explained_variance_ratio_