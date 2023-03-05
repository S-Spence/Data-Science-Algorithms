import statistics
import math
import pandas as pd
from scipy import stats

def get_test_statistic(statistic: str, observations: object) -> float:
    """
    Calculate test statistics
    params:
        statistic: the statistic to calculate
        observation: a row from a Pandas Dataframe

    """
    statistic = statistic.lower() # Ignore case
    
    if statistic == "min":
        return min(observations)
    elif statistic == "max":
        return max(observations)
    elif statistic == "mean":
        return observations.mean()
    elif statistic == "trimmed mean":
        return stats.trim_mean(observations, 0.2)
    elif statistic == "standard deviation":
        return statistics.stdev(observations)
    elif statistic == "skewness":
        return stats.skew(observations, bias=True)
    elif statistic == "kurtosis":
        # Note: Kurtosis callcs incorrect
        return stats.kurtosis(observations, fisher=False)
    else:
        raise Exception("Please enter a valid test measurement: [min, max, mean, trimmed mean, std, skewness, kurtosis] ")
    
def add_test_stats_by_class(df: object, class_labels: dict, features: list, test_statistics: list, label_name: str) -> list:
    """
    Add test statics by class and feature. Returns a nested list to use for a 
    double-indexed  pandas dataframe.
    Params:
        df: pandas dataframe
        class_labels: a dictionary storing the numerical labels that map to class types
        features: a list of all features in the dataset
        test_statistics: a list of test statistics to calculate
        label_name: the name of the df column containing the labels
    """
    statistics_by_class = []
    
    for statistic in test_statistics:
        for class_type in class_labels:
            class_list = []
            for feature in features:
                # Append stats of each feature to the class list
                feature_obs = df[df[label_name] == class_type].loc[:, feature]
                if feature != label_name:
                    calculation = round(get_test_statistic(statistic, feature_obs), 4)
                    class_list.append(calculation)
                else:
                    class_list.append(class_type)
              
            # Convert class name to class label
            class_list[-1] = class_labels[class_type]
            statistics_by_class.append(class_list)
            
    return statistics_by_class
        
def add_test_statistics_by_feature(df: object, features: list, test_statistics: list) -> list:
    """
    Add test statistics by feature
    Params:
        df: pandas dataframe
        features: a list of all features in the dataset
        test_statistics: a list of test statistics to calculate
    """
    output_list = []
    
    for statistic in test_statistics:
        stat_list = []
        for feature in features:
            stat_list.append(get_test_statistic(statistic, df[feature]))
        output_list.append(stat_list)
    return output_list 

def generate_stats_table_by_class(df: object, class_labels: dict, features: list, statistics: list, label: str, col_names: list) -> object:
    """
    Generate test statistics table by class and features
    Params:
        df: pandas dataframe
        class_labels: a dictionary storing the numerical labels that map to class types
        features: a list of all features in the dataset
        statistics: a list of test statistics to calculate
        label: the name of the df column containing the labels
        col_names: the column names for the new stats table
    """
    stats_by_class = add_test_stats_by_class(df, class_labels, features, statistics, label)
    statistics_df = pd.DataFrame(stats_by_class, 
                                 pd.MultiIndex.from_product([statistics, class_labels], 
                                 names = ["Test Statistics", "Class Labels"]), 
                                 columns = col_names)
    test_statistics_by_class = statistics_df.style.set_caption("Table 1: Test Statistics by Feature and Class")
    return test_statistics_by_class

def generate_stats_table_by_feature(df: object, features: list, statistics: list, col_names: list):
    """
    Generate test statistics table by features
    Params:
        df: pandas dataframe
        class_labels: a dictionary storing the numerical labels that map to class types
        features: a list of all features in the dataset
        statistics: a list of test statistics to calculate
        col_names: the column names for the new stats table
    """
    stats_by_feature = add_test_statistics_by_feature(df, features, statistics)
    statistics_df = pd.DataFrame(stats_by_feature, 
                                 pd.MultiIndex.from_product([statistics], 
                                 names = ["Test Statistics"]),
                                 columns = col_names)
    statistics_by_feature = statistics_df.style.set_caption("Table 2: Test Statistics by Feature")
    return statistics_by_feature
