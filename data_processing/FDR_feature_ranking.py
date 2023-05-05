"""
A file to store the feature ranking functions using Fisher's Discriminant Ratio from the
homework three assignment.
"""
import pandas as pd
import numpy as np
import sys

from data_processing import feature_ranking_helpers as ranking

sys.path.append('..')
import common_helpers as common

def get_FDR(class_1: object, class_2: object) -> list:
    """Calculate Fisher's Linear Discriminant Ratio by comparing two classes at a time"""
    mean_1 = np.mean(class_1, axis = 0)
    mean_2 = np.mean(class_2, axis= 0)
    # set ddof=1 to match matlab calculations
    variance_1 = np.var(class_1, axis=0, ddof=1)
    variance_2 = np.var(class_2, axis=0, ddof=1)
    denominator = np.array(variance_1 + variance_2)
    
    FDR = np.divide(np.square((mean_1 - mean_2)),denominator)
  
    return FDR

def FDR_two_class_feature_ranking(df: object, classes: list, label: str, ranking_method: str) -> object:
    """Generate FDR table. Pass in a dataframe without labels"""
    # split the data by class
    class_data = common.split_by_class(df, classes, label)
    # get class combinations
    class_combinations = common.get_class_combinations(classes)
    
    # Get features for column names
    features = list(df.columns)
    features.remove(label)
    
    FDRS = []
    # calculate all FDR values for class combinations
    for combination in class_combinations:
        class_1 = class_data[combination[0]].drop([label], axis=1)
        class_2 = class_data[combination[1]].drop([label], axis=1)
            
        FDR = get_FDR(class_1, class_2)   
        FDRS.append(FDR)
    
    # convert FDRs back to a dataframe
    FDR_df = pd.DataFrame(FDRS, columns = features)
    
    # create a table to store the rankings for each class combination
    row_labels = [f"{i[0]} vs. {i[1]}" for i in class_combinations] + [ranking_method]
    new_df = pd.DataFrame({"class combinations": row_labels})
    
    table, rankings = ranking.rank_features(FDR_df, ranking_method)
    
    feature_ranks = [features[val] for val in rankings]
    
    # add the row labels
    new_df = pd.concat([new_df, table.reindex(new_df.index)], axis=1)
    FDR_table = new_df.style.set_caption("FDR Feature Ranking Single Class (One vs. One)")
    
    return FDR_table, new_df, feature_ranks
    

def FDR_multiclass_feature_ranking(df: object, classes: list, label: str, ranking_method: str):
    """one vs rest"""
    # get features
    features = list(df.columns)
    features.remove(label)
    
    FDRS = []
    
    for class_type in classes:

        # Get class data and drop label
        class_data = df.loc[df[label] == class_type].drop([label], axis=1)
        # Get rest data and drop label
        rest_data = df.loc[df[label] != class_type].drop([label], axis=1)

        FDR = get_FDR(class_data, rest_data)   
        FDRS.append(FDR)
    
    # convert FDRs back to a dataframe
    FDR_df = pd.DataFrame(FDRS, columns = features)
    
    # create a table to store the rankings for each class combination
    row_labels = [f"{class_type} vs. rest" for class_type in classes] + [ranking_method]
    new_df = pd.DataFrame({"class combinations": row_labels})
    
    table, rankings = ranking.rank_features(FDR_df, ranking_method)
    
    feature_ranks = [features[val] for val in rankings]
    
    new_df = pd.concat([new_df, table.reindex(new_df.index)], axis=1)
    FDR_table = new_df.style.set_caption("FDR Feature Ranking Multiclass (One vs. Rest)")
    
    return FDR_table, new_df, feature_ranks