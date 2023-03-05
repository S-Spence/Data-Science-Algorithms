"""
K-fold Cross Validation
"""
import common_helpers as common

def get_folds(num_data: int, num_folds: int) -> dict:
    """
    Split the data into n folds and return a dictionary
    containing the list of indices for each fold.
    """
    if num_data%num_folds != 0:
        print("Please ensure your data size is divisible by the number of folds")
        return
    elif num_folds == 1:
        print("There must be at least two folds to split the data.")
        return
    else:
        indices = [i for i in range(num_data)]
        partition_size = int(num_data/num_folds)
        k_folds = {}
        
        start_indx = 0
        for fold in range(num_folds):
            k_folds[fold+1] = indices[start_indx:start_indx+partition_size]
            start_indx += partition_size
    
    return k_folds

def k_fold_validation(df: object, num_folds: int):
    """Get the test and training sets to perform k-fold cross validation"""
    # Get the shape of the data
    num_data = df.shape[0]
    # Shuffle the data
    df = common.shuffle_data(df)
    # Create a dictionary to store experiment data
    experiment_data = {}
    # Get indices for each experiment
    experiments = get_folds(num_data, num_folds)
    # Split trianing and test data at the indices for each experiment
    for experiment in experiments:
        experiment_data[experiment] = {}
        experiment_data[experiment]["test"] = df.loc[df.index[experiments[experiment]]]
        experiment_data[experiment]["train"] = df.drop(experiments[experiment], axis=0)
        
    return experiment_data