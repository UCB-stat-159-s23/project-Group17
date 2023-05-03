import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Feature choosing: 1. Remove features with low standard deviation

def prepare_features(data, columns, features_columns):
    '''
    Remove features with low standard deviation
    
    Inputs:
    - data: data matrix
    - columns: list of column names to remove
    - features_columns : list of the columns 
    Outputs:
    - x: column removed data 
    '''
    drop_features = ["person_id"] + columns

    for x in drop_features:
        features_columns.remove(x)
        
    x = data.loc[:, features_columns]
    return x


def remove_sparse_features(data, feature_r):
    '''
    Remove sparse feature
    
    Inputs:
    - data: data matrix
    - features_r : criteria of percents of zeros 
    Outputs:
    - feature_list : list of the ratios when x value equals to 0
    - x: column removed data 
    - drop_feature_sparse : columns of x datasets which feature_list is larger than feature_r
    '''
    x = data
    x_num = x.shape[0]
    feature_num = x.shape[1]
    feature_no_zero_num = np.sum([x == 0], 1)[0]
    feature_list =  feature_no_zero_num / x_num
    select_feature_sparse = x.columns[feature_list < feature_r]
    drop_feature_sparse = x.columns[feature_list > feature_r]
    x = x.loc[:, select_feature_sparse]
    return feature_list, x, drop_feature_sparse


def remove_outliers(data, f_list):
    '''
    Change the values of outliers
    
    Inputs:
    - data: data matrix
    - f_list : feature lists that you wanna trim or edit
    Outputs:
    - data : processed data after changing the values of outliers
    '''

    for feature in f_list:
        feature_data = data[feature]
        mean = data[feature].mean()
        std = data[feature].std()
        mean_3_std = mean + 3 * std
        data.loc[data[feature] > mean_3_std, feature] = mean_3_std
    return data