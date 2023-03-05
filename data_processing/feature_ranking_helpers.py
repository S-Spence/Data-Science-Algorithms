"""
Functions used by multiple feature ranking algorithms
"""

def rank_features(table: object, method: str):
    """
    Ranking Methods: 
        - min: take the minimum value of each combination
        - max: take the maximum value of each combination
        - sum: take the sum of all combinations
        - avg: take the average of all combinations
        
    The values will then be taken in descending order to rank the features
    """
    row = []
    if method == "sum":
        # calculate features sums and add as a new row
        row = list(table.iloc[:, :].sum(axis=0))
        table.loc[len(table.index)] = row
    elif method == "min":
        row = list(table.iloc[:, :].min(axis=0))
        table.loc[len(table.index)] = row
    elif method == "max":
        row = list(table.iloc[:, :].max(axis=0))
        table.loc[len(table.index)] = row
    elif method == "avg":
        mins = list(table.iloc[:, :].min(axis=0))
        maxs = list(table.iloc[:, :].max(axis=0))
        row = [mins[i] + maxs[i]/2 for i in range(len(mins))]
        table.loc[len(table.index)] = row
    else:
        print("Please use a valid ranking method: (sum, min, max, avg)")
        return table, None
        
    # Track indices
    row_indices = {}
    for indx, val in enumerate(row):
        if val in row_indices.keys():
            row_indices[val].append(indx)
        else:
            row_indices[val] = [indx]
        
    sorted_vals = sorted(row, reverse=True)
    rankings = []
    rank_values = []
    
    # append as lists incase there were duplicate rank values
    for val in sorted_vals:
        rank_values.append(val)
        rankings.append(row_indices[val])
    
    # unpack lists for final output
    final_rankings = []
    for array in rankings:
        for val in array:
            final_rankings.append(val)
        
    return table, final_rankings