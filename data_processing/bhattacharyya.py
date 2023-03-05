"""
The bhattacharyya distance measures the similarity of two probability distributions. The bhattachryaa distance can be users for feature ranking and outlier removal.
"""
import pandas as pd
import numpy as np

from data_processing import feature_ranking_helpers as ranking
import common_helpers as common

def bhattacharyya(data: dict):
    """Calculate the Bhattacharyya distances of the data"""
    x = data["x"]
    y = data["y"]
    
    mean_1_indices = np.where(y == -1)
    cutoff = len(mean_1_indices[0])
    
    lower_matrix = x[:cutoff, :]
    upper_matrix = x[cutoff:, :]
    
    mean_1 = np.mean(lower_matrix, axis=0)
    mean_2 = np.mean(upper_matrix, axis=0)

    cov_1 = np.cov(lower_matrix, rowvar=False)
    cov_2 = np.cov(upper_matrix, rowvar=False)
    
    B1_1 = np.multiply((1/8), mean_1 - mean_2)
    B1_2 = np.diag(np.linalg.inv(np.divide(cov_1+cov_2,2))).T
    B1_3 = np.multiply(B1_1, B1_2)
    B1 = np.multiply(B1_3, mean_1-mean_2)
    
    B2_1 = np.diag(np.divide((cov_1 + cov_2), 2)).T
    B2_2 = np.divide(B2_1, np.sqrt(np.multiply(np.abs(np.diag(cov_1).T), np.abs(np.diag(cov_2).T))))
    B2 = np.multiply(1/2, np.log(B2_2))
                       
    B = B1 + B2
    
    return B

def bhattacharrya_multiclass_feature_ranking(df: object, classes: list, label: str, ranking_method: str):
    """one vs rest"""
    # get features
    features = list(df.columns)
    features.remove(label)
    
    bhatta_rankings = []
    
    for class_type in classes:
        
        data = {"x": [], "y": []}

        # drop the labels and convert to a numpy array 
        class_data = df.loc[df[label] == class_type]
        num_class_data = len(class_data.index)

        rest_data = df.loc[df[label] != class_type]
        num_rest_data = len(rest_data.index)

        new_data = pd.concat([class_data, rest_data]).drop([label], axis=1)
        data["x"] =  np.array(new_data)

        # set the first class to -1 and the second class to 1
        y1 = [-1 for i in range(num_class_data)]
        y2 = [1 for i in range(num_rest_data)]

        y = y1 + y2
        data["y"] = np.array(y)

        B = bhattacharyya(data)

        bhatta_rankings.append(B)
        
    # create a table to store the rankings for each class combination
    row_labels = [f"{class_type} vs. rest" for class_type in classes] + [ranking_method]
    new_df = pd.DataFrame({"class combinations": row_labels})
    
    # convert values back to a dataframe
    bhatta_df = pd.DataFrame(bhatta_rankings, columns = features)
    
    table, rankings = ranking.rank_features(bhatta_df, ranking_method)
    
    feature_ranks = [features[val] for val in rankings]
    # add the row labels
    new_df = pd.concat([new_df, table.reindex(new_df.index)], axis=1)
    bhatta_table = new_df.style.set_caption("Bhattacharyya Feature Ranking Multiclass (One vs. Rest)")
    
    return bhatta_table, new_df, feature_ranks
        
            
def bhattacharyya_two_class_feature_ranking(df: object, classes: list, label: str, ranking_method:str):
    """
    Use the Bhattacharyya distance for feature ranking.
    The bhattacharyya distance measures the separability of two classes.
    Ranking Methods: 
        - min: take the minimum value of each combination
        - max: take the maximum value of each combination
        - sum: take the sum of all combinations
        - avg: take the average of all combinations
        
    """
    # split the data by class
    class_data = common.split_by_class(df, classes, label)
    # get class combinations
    class_combinations = common.get_class_combinations(classes)
    # get features
    features = list(df.columns)
    features.remove(label)
    
    bhatta_rankings = []
        
    for combination in class_combinations:

        data = {"x": [], "y": []}

        # drop the labels and convert to a numpy array 
        class_1 = class_data[combination[0]]
        num_class_1 = len(class_1.index)

        class_2 = class_data[combination[1]]
        num_class_2 = len(class_2.index)

        new_data = pd.concat([class_1, class_2]).drop([label], axis=1)
        data["x"] =  np.array(new_data)

        # set the first class to -1 and the second class to 1
        y1 = [-1 for i in range(num_class_1)]
        y2 = [1 for i in range(num_class_2)]

        y = y1 + y2
        data["y"] = np.array(y)

        B = bhattacharyya(data)

        bhatta_rankings.append(B)
        
    
    # create a table to store the rankings for each class combination
    row_labels = [f"{i[0]} vs. {i[1]}" for i in class_combinations] + [ranking_method]
    new_df = pd.DataFrame({"class combinations": row_labels})
    
    # convert values back to a dataframe
    bhatta_df = pd.DataFrame(bhatta_rankings, columns = features)
    
    table, rankings = ranking.rank_features(bhatta_df, ranking_method)
    
    feature_ranks = [features[val] for val in rankings]
    
    # add the row labels
    new_df = pd.concat([new_df, table.reindex(new_df.index)], axis=1)
    bhatta_table = new_df.style.set_caption("Bhattacharyya Feature Ranking Single Class (One vs. One)")
    
    return bhatta_table, new_df, feature_ranks