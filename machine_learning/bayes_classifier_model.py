import common_helpers as common
import numpy as np
import pandas as pd

class PClass(): 
    def __init__(self, prior=[], mean=[], cov=[]):
        self.prior = prior
        self.mean = mean
        self.cov = cov
    
    def set_prior(self, prior):
        self.prior = prior
    
    def set_mean(self, mean):
        self.mean = mean
        
    def set_cov(self, cov):
        self.cov = cov
    
    
class Model: 
    def __init__(self, p_classes=[], prior=[]):
        self.p_class = p_classes
        self.prior = prior
        
    def set_prior(self, prior: list):
        self.prior = prior
    
    def add_class(self, p_class: object):
        self.p_class.append(p_class)
    
    def set_pclasses(self, p_classes):
        self.p_class = p_classes
    
    
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