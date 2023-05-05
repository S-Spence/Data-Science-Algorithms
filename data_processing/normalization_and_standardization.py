def normalize(df: object):
    """
    Normalize the data passed in using min/max normalization to
    put the data in a range from 0-1. Normalization is best when
    the data contains no outliers.
    """
    norm_df = df.copy()
  
    # Apply min/max normalization for each column
    for col in norm_df.columns:
        norm_df[col] = (norm_df[col] - norm_df[col].min())/(norm_df[col].max() - norm_df[col].min()) 
    return norm_df


def standardize(df: object):
    """
    Standardize the data passed in by giving it a standard
    deviation of 0 and a mean of 1. Standardizatio  is useful
    when the data has a Gaussian distribution. Standardization 
    is not impacted by outliers like normalization.
    """
    return (df-df.mean())/df.std()