# Bayes Classification functions
import sys
import numpy as np
import math

import common_helpers as common
from machine_learning import k_fold_validation as k_fold

# append the path of theparent directory
sys.path.append("..")
 
# import method from sibling
# module
from data_processing import mahalanobis as mahal
from machine_learning import bayes_classifier_model as bayes_model


def build_model(data: object, classes: list, label: str):
    """Build a model for the bayes classifier"""
    model = Model()
    
    classes = common.split_by_class(data, classes, label)
    prior = []
    for class_type, values in classes.items():
        prior.append(len(values)/len(data))
    
    model.set_prior(prior)
    
    p_classes = []
    for class_type, values in classes.items():
        values = values.drop([label], axis=1)
        values = np.array(values)
        class_prior = 1
        class_mean = np.mean(values, axis = 0)
        class_mean = np.reshape(class_mean, (len(class_mean), 1))
        # Pass the data transpose to numpy.cov to get it to match matlab
        class_cov = np.cov(values.T)
        
        p_class = PClass(class_prior, class_mean, class_cov)
        
        p_classes.append(p_class)
    
    model.set_pclasses(p_classes)
        
    return model

def multivariate_gaussian(data, mean, cov):
    """Evaluate a multivariate Gaussian"""
    num_observations, dimensions = np.shape(data)
    _, ncomp = np.shape(mean)
    
    y = np.zeros([ncomp, num_observations])
    
    for i in range(ncomp):
        dist = mahal.mahalanobis_with_model(data.T, mean[:, i], cov)
        calc_1 = np.exp(np.multiply(-0.5, dist))
        calc_2 = np.power(2 * math.pi, dimensions)
        calc_3 = np.linalg.det(cov)
        calc_4 = math.sqrt(calc_2 * calc_3)
        y[i, :] = np.divide(calc_1, calc_4)
    return y
    
def bayes_classifier(data: object, model: object):
    """Bayes Classifier algorithm. Note: data must be normalized."""
    observations, dimensions = np.shape(data)
    num_classes = len(model.p_class)
    y_posterior = np.zeros([num_classes, observations])
    
    for i in range(num_classes):
        gaussian = multivariate_gaussian(data, model.p_class[i].mean, model.p_class[i].cov)
        y = np.multiply(model.p_class[i].prior, gaussian)
        y_posterior[i, :] = model.prior[i] * y
 
    y_pred = list(y_posterior.argmax(axis=0))
    
    return y_pred, y_posterior

def bayes_classifier_k_fold(df: object, num_folds:int, classes: list, label_name: str):
    """Use bayes classification with k-fold validation. Note: classes should be converted to numbers"""
    accuracy_totals = {}
    experiments = k_fold.k_fold_validation(df, num_folds)

    for experiment in experiments:
        # get test data and convert labels to number
        test_df = experiments[experiment]["test"]
        test_labels = np.array(test_df["label"])
        data = test_df.drop([label_name], axis=1)
        test_data = np.array(data)
        # get training data 
        train_df = experiments[experiment]["train"]
        train_labels = np.array(train_df["label"])
        data = train_df.drop([label_name], axis=1)
        train_data = np.array(data)
        
        # Fit the model with the training dataset
        model = bayes_model.build_model(train_df, classes, label_name)

        # Determine classification accuracy with the test dataset
        classifications, y_posterior = bayes_classifier(test_data, model)
        accuracy = common.get_accuracy(classifications, test_labels)
        accuracy_totals[experiment] = accuracy
        print(f"Experiment {experiment}:\n")
        print(f"The Bayes classifier had an accuracy of {accuracy}% on the test data.")
        common.confusion_matrix(test_labels, classifications)
        
    print(f"The average accuracy among the experiments was {round(sum(accuracy_totals.values())/len(accuracy_totals.values()), 2)}%")
