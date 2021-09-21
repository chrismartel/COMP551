# Author: Christian Martel, Luka Loignon, Marie Guertin
# Date: 2021-09-20

import pandas as pd

def ohe(df):
    '''One-hot-encoding of categorical columns of a dataframe'''

    for c in df.columns:    
        if pd.api.types.is_object_dtype(df[c]):
            # concatenante one hot encoded dummies columns
            df = pd.concat([df, pd.get_dummies(df[c], prefix = c, prefix_sep = '_')], axis = 1)

            # drop old column
            df.drop(columns = c, inplace=True)
    return df

def imputation(df, invalid_entries_dict = dict()):
    '''Imputation of NaN values or other invalid entries of a dataframe.
       For continuous features: replace by mean value
       For categorical features: replace by most occurring value
       
       invalid_entries is a dict of lists. The keys are the column
       names and the values are the list of invalid entries for each column.'''

    for c in invalid_entries_dict.keys():
        # step 1: fill NaN values

        # continuous feature
        if pd.api.types.is_numeric_dtype(df[c]):
            # fill with mean
            df[c].fillna(df[c].mean(), inplace=True)

        # categorical feature
        else:
            # fill with most occurring value
            df[c].fillna(df[c].mode(), inplace=True)
        
        # step 2: fill invalid_entries
        for entry in invalid_entries_dict[c]:
            df[c].replace(to_replace= entry, value=df[c].mode().iloc[0], inplace=True)
    return df

def strip_labels(df):
    '''Strip labels of categorical data in dataframe'''

    df_obj = df.select_dtypes(['object'])

    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    return df


