"""
Michael Pletcher
Created: 04/30/2025
Edited: 05/01/2025

##### Summary #####
.py script containing functions for training
and testing machine learning models for snow-to-liquid
ratio prediction.
"""

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

def build_train_test_data(
    data,
    label = 'slr',
    train_size = 0.6,
    test_size = 0.4,
    strata_ = True,
    strata_factor = 10,
):
    """
    Build train and test datasets

    Parameters:
    data : pd.DataFrame
        pandas Dataframe containing features and labels to
        be used for the train and test datasets
    label : str
        Name of column in data to be used as target
    train_size : float
        Fraction of data to be used for training
    test_size : float
        Fraction of data to be used for testing
    strata_ : Boolean, default is True
        Stratifies data into classes to ensure even
        distributions of labels in train and test datasets
    strata_factor : int, default is 10
        Range of classes to be grouped (e.g., 1-10, 11-20, etc.)

    Returns:
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features
    y_train : pd.DataFrame
        Training labels aligned with X_train
    y_test : pd.DataFrame
        Testing labels aligned with X_test
    """

    strata = np.round(data[label] / strata_factor, 0) * strata_factor
    n_events_smallest_class = strata.value_counts().min()
    if n_events_smallest_class < 2:
        print(
            'Warning : Stratify not possible, smallest class contains %s event(s). Training without stratifying' % n_events_smallest_class
        )
    
    if strata_:
        X_train, X_test = train_test_split(
            data,
            train_size = train_size,
            test_size = test_size,
            stratify = strata if data[label].value_counts().min() >= 2 else None
        )
    else:
        X_train, X_test = train_test_split(
            data,
            train_size = train_size,
            test_size = test_size,
        )
    
    y_train, y_test = X_train[label], X_test[label]

    # Clean up and match
    X_train, X_test = X_train.reset_index(), X_test.reset_index()
    X_train, X_test = X_train.dropna(), X_test.dropna()
    y_train, y_test = y_train.iloc[X_train.index.values], y_test.iloc[X_test.index.values]
    X_train, X_test = X_train.drop(columns = ['index']), X_test.drop(columns = ['index'])

    return X_train, X_test, y_train, y_test

def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    features = None,
    model_type = 'rf',
    ntrees = 100
):
    """
    Train machine learning models using inputs train and test datasets

    Parameters:
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features
    y_train : pd.DataFrame
        Training labels aligned with X_train
    y_test : pd.DataFrame
        Testing labels aligned with X_test
    features : list of str
        Column names of features
    model_type : str
        Type of ML model to train
    ntrees : int
        If model_type == 'RF', specifies
        number of trees in the random forest

    Returns:
    test_predictions : np.array
        Predictions made by ML model
    y_test : np.array
        Labels in test dataset
        
    """
    # Save these data to be returned later as X_train and X_test
    lat, lon, elev, site = X_test.lat, X_test.lon, X_test.elev, X_test.site

    X_train, X_test = X_train[features], X_test[features]

    start = time.perf_counter()

    # Clean up
    y_train = y_train.reset_index()
    y_train = y_train.drop('datetime_utc', axis = 1)
    X_train, y_train = X_train.dropna(), y_train.dropna()

    # Random forest
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators = ntrees,
            min_samples_split = 2,
            min_samples_leaf = 2,
            max_features = 'sqrt',
            max_depth = 100,
            bootstrap = True
        ).fit(X_train, y_train)
    # Linear regression
    elif model_type == 'lr':
        model = LinearRegression().fit(X_train, y_train)

    # Predict
    test_predictions = model.predict(X_test)
    end = time.perf_counter()

    elapsed = end - start
    print('Elapsed time to train and make predictions: %.2f' % elapsed)

    if model_type == 'lr':
        return np.array(test_predictions[:, 0]), np.array(y_test), model
    else:
        return np.array(test_predictions), np.array(y_test), model

def evaluate_model(y_test, test_predictions):
    """
    Evaluate ML models with several statistics

    Parameters:
    y_test : np.array
        Test dataset labels
    test_predictions : np.array
        Test predictions matched with test labels

    Returns:
    metrics : dict of np.array
        Mean absolute error (MAE), root mean squared error (RMSE),
        coefficient of determination (R2; from scikit-learn),
        and mean bias error (MBE)
    """

    metrics = {
        "MAE" : np.nanmean(abs(test_predictions - y_test)),
        'RMSE' : np.sqrt(np.nanmean((test_predictions - y_test) ** 2)),
        'R2' : r2_score(y_test, test_predictions),
        'MBE' : np.nanmean(test_predictions - y_test),
    }
    return metrics