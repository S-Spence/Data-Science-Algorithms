"""
Parzen window algorithm
"""
import numpy as np
import math

import common_helpers as common
from machine_learning import k_fold_validation as k_fold

def gaussian_kernel(test_data: object, training_data: object, spread: float):
    """
    Evaluate the Gaussian of the input data and return the kernel values
    Note: data must be normalized
    """
    # Get the shape of the test data
    test_data = np.array(test_data)
    test_rows, test_cols = np.shape(test_data)
 
    # Get the shape of the training data
    training_data = np.array(training_data)
    training_rows, training_cols = np.shape(training_data)

    kernels = np.zeros((test_rows, training_rows))
    
    for i in range(test_rows):
        for j in range(training_rows):
            calc_1 = (1/(math.pow((np.sqrt(2 * math.pi) * spread), training_cols)))
            calc_2 = np.exp(-0.5 * ((np.matmul((test_data[i, :]-training_data[j, :]), (test_data[i, :] - training_data[j, :]).T)/(math.pow(spread, 2)))))
            kernels[i, j] = calc_1 * calc_2
    
    return kernels

def create_class_window(train_data: object, test_data: object, train_labels: object, test_labels: object, class_type: str, spread: float):
    """Create class window"""
    train = []
    for i in range(len(train_labels)):
        if train_labels[i] == class_type:
            train.append(train_data[i, :])
    
    parzen_window = np.zeros((len(test_data), 1))
    for i in range(len(test_data)):
        test_example = np.reshape(test_data[i, :], (1, np.shape(test_data[i, :])[0]))
        kernels = gaussian_kernel(test_example, train, spread)
        sum_kernels = np.sum(kernels)
        parzen_window[i] = (1/len(train))*sum_kernels
    return parzen_window

def run_parzen_window(train_df: object, train_labels: object, test_df: object, test_labels:object, classes: list, spread:float):
    """Use the parzen window approach to perform classification. Class types must be numerical."""
    accuracy = {}
    parzen_windows = np.array([])
    
    # convert to numpy arrays 
    train_data = np.array(train_df)
    test_data = np.array(test_df)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    print("Parzen window training...\n")
    for class_type in classes:
        print(f"Training with class {class_type}/{classes[-1]}")
        parzen = create_class_window(train_data, test_data, train_labels, test_labels, class_type, spread)
        parzen_windows = common.stack_columns(parzen_windows, parzen)
  
    parzen_windows = parzen_windows.T
    y_pred = list(parzen_windows.argmax(axis=0))  
    correct_pred = 0
    for i, val in enumerate(y_pred):
        if y_pred[i] == test_labels[i]:
            correct_pred += 1
            
    accuracy = (correct_pred/len(test_labels))*100;
    print(f"\nAccuracy: {accuracy}%")
        
    common.confusion_matrix(test_labels, y_pred)
        
    return parzen_windows

def parzen_window_k_fold(df: object, num_folds:int, classes: list, label_name: str, spread:float):
    """Use the parzen window approach to perform classification"""
    experiments = k_fold.k_fold_validation(df, num_folds)
    accuracy_totals = {}
    for experiment in experiments:
        # get test data and convert labels to number
        test_df = experiments[experiment]["test"]
        test_labels = np.array(test_df[label_name])
        data = test_df.drop([label_name], axis=1)
        test_data = np.array(data)
        # get training data and convert labels to numbers
        train_df = experiments[experiment]["train"]
        train_labels = np.array(train_df[label_name])
        data = train_df.drop([label_name], axis=1)
        train_data = np.array(data)
        
        parzens = np.array([])
        for class_type in classes:
            new_parzen = create_class_window(train_data, test_data, train_labels, test_labels, class_type, spread)
            parzens = common.stack_columns(parzens, new_parzen)
        
        parzens = parzens.T
        
        y_pred = list(parzens.argmax(axis=0))
        
        accuracy = common.get_accuracy(y_pred, test_labels)
        accuracy_totals[experiment] = accuracy
        print(f"Experiment {experiment} accuracy: {accuracy}%")
        
        common.confusion_matrix(test_labels, y_pred)
    print(f"The average accuracy among the experiments was {round(sum(accuracy_totals.values())/len(accuracy_totals), 2)}%")
    return parzens