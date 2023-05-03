import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from tools import utils

def test_prepare_feature():
    data = pd.DataFrame({
        "person_id": [1, 2, 3],
        "col1": [0.5, 0.6, 0.7],
        "col2": [1.0, 2.0, 3.0],
        "col3": [0.1, 0.2, 0.3],
        "col4": [5.0, 5.0, 5.0]
    })
    columns = ["col4"]
    features_columns = list(data.columns)
    expected_output = pd.DataFrame({
        "col1": [0.5, 0.6, 0.7],
        "col2": [1.0, 2.0, 3.0],
        "col3": [0.1, 0.2, 0.3]
    })
    assert utils.prepare_features(data, columns, features_columns).equals(expected_output)

def test_remove_sparse_features():
    data = pd.DataFrame({'a': [1, 2, 3], 'b': [0, 0, 0], 'c': [0, 4, 5]})
    feature_r = 0.5
    
    feature_list, x, drop_feature_sparse = utils.remove_sparse_features(data, feature_r)
    
    assert np.array_equal(x.columns, np.array(['a','c']))
    assert np.array_equal(drop_feature_sparse, np.array(['b']))
    
def test_remove_outliers():
    
    test_df = pd.DataFrame({
        'feature_1': [100, 200, 300, 400, 500],
        'feature_2': [1, 2, 3, 4, 100],
        'feature_3': [10, 20, 30, 40, 50]
    })
    
    f_list = ['feature_1', 'feature_2']
    processed_df = utils.remove_outliers(test_df, f_list)
    
    assert processed_df.iloc[0]['feature_1'] == 100.0
    