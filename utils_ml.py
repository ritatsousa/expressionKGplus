import requests
import pickle
import csv
import statistics
import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ParameterGrid


def write_metrics(path_output_file, list_metrics, list_values):
    """
    Write metrics and their corresponding values to a tab-separated values (TSV) file.
    Parameters:
        path_output_file (str): The path to the output file where metrics will be saved.
        list_metrics (list of str): List of metric names.
        list_values (dict): Dictionary with values corresponding to each metric.
    Returns:
        None
    """
    with open(path_output_file, "w") as output_file:
        output_file.write('Metric\tValue\n')
        for i in range(len(list_metrics)):
            output_file.write(list_metrics[i] + '\t' + str(list_values[i]) + '\n')


def compute_metrics(y, y_pred):
    """
    Compute a variety of performance metrics for classification predictions.
    Parameters:
        y (array-like): True labels.
        y_pred (array-like): Predicted labels.
    Returns:
        pr, rec, f1, acc, waf (tuple): A tuple containing the following metrics in order Precision (float), Recall (float), F1 score (float), Accuracy (float), Weighted F1 score (float)
    """
    pr = metrics.precision_score(y, y_pred) 
    rec = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred) 
    acc = metrics.accuracy_score(y, y_pred)  
    waf = metrics.f1_score(y, y_pred, average="weighted")
    return pr, rec, f1, acc, waf


def compute_metrics_with_proba(y, y_pred, y_proba):
    """
    Compute a variety of performance metrics for classification predictions, including AUC.
    Parameters:
        y (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like): Predicted probabilities for the positive class.
    Returns:
        pr, rec, f1, acc, waf, auc (tuple): A tuple containing the following metrics in order: Precision (float), Recall (float), F1 score (float), Accuracy (float), Weighted F1 score (float), AUC (float)
    """
    pr, rec, f1, acc, waf = compute_metrics(y, y_pred)
    auc = metrics.roc_auc_score(y, y_proba)
    return pr, rec, f1, acc, waf, auc


def write_predictions(path_metrics, y, y_pred):
    """
    Write the true labels and predicted labels to a tab-separated values (TSV) file.
    Parameters:
        path_metrics (str): The path to the output file where predictions will be saved.
        y (array-like): True labels.
        y_pred (array-like): Predicted labels.
    Returns:
        None
    """
    with open(path_metrics, 'w') as file_pred:
        file_pred.write("Pred\tLabel\n")
        y, y_pred = list(y), list(y_pred)
        for i in range(len(y)):
            file_pred.write(str(y_pred[i])+"\t"+str(y[i])+"\n")


def store_metrics(dic_metrics, list_metrics, list_values):
    """
    Update a dictionary with lists of metric values.
    Parameters:
        dic_metrics (dict): Dictionary where keys are metric names and values are lists of metric values.
        list_metrics (list of str): List of metric names.
        list_values (list of floats): List of values corresponding to each metric.
    Returns:
        dic_metrics (dict): Updated dictionary with appended metric values.
    """
    for i in range(len(list_metrics)):
        dic_metrics[list_metrics[i]] = dic_metrics[list_metrics[i]] + [list_values[i]]
    return dic_metrics


def train_ml_model(X_train, X_test, y_train, y_test, alg, **params):
    """
    Train a ML model and evaluate its performance.
    Parameters:
        X_train (array-like): Training feature data.
        X_test (array-like): Test feature data.
        y_train (array-like): Training labels.
        y_test (array-like): Test labels.
        alg (str): The algorithm to use for training the model ("MLP" for Multi-Layer Perceptron).
        **params: Additional parameters to pass to the model's constructor.
    Returns:
        clf, pr_test, rec_test, f1_test, acc_test, waf_test, auc_test (tuple): A tuple containing the trained classifier and the following test performance metrics: Precision (float), Recall (float), F1 score (float), Accuracy (float), Weighted F1 score (float), AUC (float)
    """
    if alg == "MLP":
        clf = MLPClassifier(**params)

    # Train the classifier to the training data
    clf.fit(X_train, y_train)
    # Predict the labels for the test data
    y_pred = clf.predict(X_test)

    # Compute performance metrics
    pr_test, rec_test, f1_test, acc_test, waf_test, auc_test = compute_metrics_with_proba(y_test, y_pred, clf.predict_proba(X_test)[:,1])
    return clf, pr_test, rec_test, f1_test, acc_test, waf_test, auc_test


def train_ml_model_complete(path_metrics, X_train, X_test, y_train, y_test, alg, **params):
    """
    Train a machine learning model, evaluate its performance, and save predictions.
    Parameters:
        path_metrics (str): The path where prediction files will be saved.
        X_train (array-like): Training feature data.
        X_test (array-like): Test feature data.
        y_train (array-like): Training labels.
        y_test (array-like): Test labels.
        alg (str): The algorithm to use for training the model ("MLP" for Multi-Layer Perceptron).
        **params: Additional parameters to pass to the model's constructor.
    Returns:
        clf, pr_test, rec_test, f1_test, acc_test, waf_test, auc_test, pr_train, rec_train, f1_train, acc_train, waf_train, auc_train (tuple): 
        A tuple containing the trained classifier and the following test and training performance metrics:
        - Precision (float) for test data
        - Recall (float) for test data
        - F1 score (float) for test data
        - Accuracy (float) for test data
        - Weighted F1 score (float) for test data
        - AUC (float) for test data
        - Precision (float) for training data
        - Recall (float) for training data
        - F1 score (float) for training data
        - Accuracy (float) for training data
        - Weighted F1 score (float) for training data
        - AUC (float) for training data
    """
    if alg == "MLP":
        clf = MLPClassifier(**params)

    # Train the classifier to the training data
    clf.fit(X_train, y_train)
    # Predict the labels for the test and training data
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    # Compute performance metrics for the test data
    pr_test, rec_test, f1_test, acc_test, waf_test, auc_test = compute_metrics_with_proba(y_test, y_pred, clf.predict_proba(X_test)[:,1])
    # Compute performance metrics for the training data
    pr_train, rec_train, f1_train, acc_train, waf_train, auc_train = compute_metrics_with_proba(y_train, y_pred_train, clf.predict_proba(X_train)[:,1])

    # Write predictions to files
    write_predictions(path_metrics + 'PredictionsTrain.tsv', y_train, y_pred_train)
    write_predictions(path_metrics + 'PredictionsTest.tsv', y_test, y_pred)

    return clf, pr_test, rec_test, f1_test, acc_test, waf_test, auc_test, pr_train, rec_train, f1_train, acc_train, waf_train, auc_train


def run_ml_model_one_partition(param_grid, optimization, cv, path_metrics, alg, metrics_cross_cv, X_train, X_valid, X_train_valid, X_test, y_train, y_valid, y_train_valid, y_test):
    """
    Run a machine learning model for one partition of data, train the model, and save results.
    Parameters:
        param_grid (dict): The parameter grid for model parameter optimization.
        optimization (bool): Whether to perform parameter optimization.
        cv (int): Cross-validation fold number.
        path_metrics (str): The path where metric files will be saved.
        alg (str): The algorithm to use for training the model ("MLP" for Multi-Layer Perceptron).
        metrics_cross_cv (dict): A dictionary to store cross-validation metrics.
        X_train (array-like): Training feature data.
        X_valid (array-like): Validation feature data.
        X_train_valid (array-like): Combined training and validation feature data.
        X_test (array-like): Test feature data.
        y_train (array-like): Training labels.
        y_valid (array-like): Validation labels.
        y_train_valid (array-like): Combined training and validation labels.
        y_test (array-like): Test labels.
    Returns:
        metrics_cross_cv (dict): Updated metrics_cross_cv dictionary with the latest cross-validation metrics.
    """
    param_combinations = ParameterGrid(param_grid)
    best_waf = 0
    best_params = {}
        
    if optimization:
        # Perform parameter optimization
        for params in param_combinations:
            clf, pr, rec, f1, acc, waf, auc = train_ml_model(X_train, X_valid, y_train, y_valid, alg, **params)
            if waf > best_waf:
                best_waf = waf
                best_params = params
    else:
        # Select the first parameter combination if no optimization
        for params in param_combinations:
            best_params = params

    # Save the best parameters to a file
    with open(path_metrics + str(cv) + '_Best_parameters.txt', 'w') as file_parameters:
        file_parameters.write(str(best_params))

    # Train the model with the best parameters and evaluate its performance
    clf, pr, rec, f1, acc, waf, auc, pr_train, rec_train, f1_train, acc_train, waf_train, auc_train =  train_ml_model_complete(path_metrics+ str(cv) + '_', X_train_valid, X_test, y_train_valid, y_test, alg, **best_params)
    print('training and predicting ML model - DONE!!')

    # Save the trained model to a file    
    with open(path_metrics + str(cv) + '_Model.pickle', 'wb') as file_clf:
        pickle.dump(clf, file_clf)
    print('saving ML model - DONE!!')

    # Write test and training metrics to files
    write_metrics(path_metrics + str(cv) + '_MetricsTest.tsv', ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'WAF', 'AUC'], [acc, pr, rec, f1, waf, auc]) 
    write_metrics(path_metrics + str(cv) + '_MetricsTrain.tsv', ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'WAF', 'AUC'], [acc_train, pr_train, rec_train, f1_train, waf_train, auc_train])

    return store_metrics(metrics_cross_cv, ["acc", "pr", "rec", "f1", "waf", "auc"], [acc, pr, rec, f1, waf, auc])


def get_parameter_grid_mlp(optimization):
    """
    Get the parameter grid for training an MLPClassifier.
    Parameters:
        optimization (bool): Whether to include a wide range of parameter values for optimization.
    Returns:
        param_grid (dict): A dictionary representing the parameter grid for the MLPClassifier.
    """
    if optimization:
        # Parameter grid for optimization
        param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (30, 20, 10)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        }
    else:
        # Default parameter grid
        param_grid = {
        'hidden_layer_sizes': [(100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'learning_rate': ['constant'],
        }
    return param_grid

