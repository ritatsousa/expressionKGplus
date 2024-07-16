import statistics
import pandas as pd
import pickle
import numpy as np
import random

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ParameterGrid

from utils_datasets import process_labels_file, read_features_file
from utils_ml import write_metrics, get_parameter_grid_mlp, run_ml_model_one_partition
from utils_cv import extract_train_test_valid_entities, extract_X_train_test_valid, extract_y_train_test_valid

import os


def run_ml_model(path_features_file, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities, path_valid_entities, optimization):
    """
    Run a ML model across specified cross-validation folds.
    Parameters:
        path_features_file (str): Path to the features file.
        path_metrics (str): Path where metrics will be saved.
        cv_folds (int): Number of cross-validation folds.
        alg (str): The ML algorithm to use (e.g., "MLP").
        path_entities_label (str): Path to the entities labels file.
        path_train_entities (str): Path to the training entities files.
        path_test_entities (str): Path to the testing entities files.
        path_valid_entities (str): Path to the validation entities files.
        optimization (bool): Whether to optimize hyperparameters.
    Returns:
        None
    """
    # Processing labels file
    dic_labels = process_labels_file(path_entities_label)

    # Read features from the provided features file
    dic_features = read_features_file(path_features_file)

    # Get the parameter grid for the specified algorithm
    if alg == "MLP":
        param_grid = get_parameter_grid_mlp(optimization)

    metrics_cross_cv = {"pr":[], "rec":[], "f1":[], "acc":[], "waf":[], "auc":[]}

    # Loop through each cross-validation fold
    for cv in range(cv_folds):
        # Extract train, valid, and test entities for the current fold
        train_entities, valid_entities, train_valid_entities, test_entities = extract_train_test_valid_entities(path_train_entities, path_valid_entities, path_test_entities, cv)
        # Extract feature matrices and labels for training, validation, and testing
        X_train, X_valid, X_train_valid, X_test = extract_X_train_test_valid(dic_features, train_entities, valid_entities, train_valid_entities, test_entities)
        y_train, y_valid, y_train_valid, y_test = extract_y_train_test_valid(dic_labels, train_entities, valid_entities, train_valid_entities, test_entities)
        # Run the ML model for the current partition
        metrics_cross_cv = run_ml_model_one_partition(param_grid, optimization, cv, path_metrics, alg, metrics_cross_cv, X_train, X_valid, X_train_valid, X_test, y_train, y_valid, y_train_valid, y_test)

    # Calculate average metrics across all folds
    metric_values = [statistics.mean(metrics_cross_cv["acc"]), statistics.mean(metrics_cross_cv["pr"]), statistics.mean(metrics_cross_cv["rec"]), statistics.mean(metrics_cross_cv["f1"]),statistics.mean(metrics_cross_cv["waf"]), statistics.mean(metrics_cross_cv["auc"])]
    # Write the average metrics to a specified output file
    write_metrics(path_metrics + 'AverageMetrics.tsv', ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'WAF', 'AUC'], metric_values)



def run_ml_model_transfer_learning(path_features_file, path_metrics, alg, path_entities_label,  path_entities_dataset1, path_entities_dataset2, path_entities_dataset3, optimization): 
    """
    Run a ML model using transfer learning across specified datasets.
    Parameters:
        path_features_file (str): Path to the features file.
        path_metrics (str): Path where metrics will be saved.
        alg (str): The ML algorithm to use (e.g., "MLP").
        path_entities_label (str): Path to the entities labels file.
        path_entities_dataset1 (str): Path to the first dataset's entities file (for training).
        path_entities_dataset2 (str): Path to the second dataset's entities file (for training).
        path_entities_dataset3 (str): Path to the third dataset's entities file (for testing).
        optimization (bool): Whether to optimize hyperparameters.
    Returns:
        None
    """
    # Processing labels file
    dic_labels = process_labels_file(path_entities_label)

    # Read features from the provided features file
    dic_features = read_features_file(path_features_file)

    # Extract entities from the datasets
    dataset1_entities = [ent.strip().split("\t")[0] for ent in open(path_entities_dataset1, 'r').readlines()]
    dataset2_entities = [ent.strip().split("\t")[0] for ent in open(path_entities_dataset2, 'r').readlines()]
    
    # Combine entities from the first two datasets for training
    train_entities = dataset1_entities + dataset2_entities
    # Extract test entities from the third dataset
    test_entities = [ent.strip().split("\t")[0] for ent in open(path_entities_dataset3, 'r').readlines()]
    print('processing test and train files - DONE!!')

    # Prepare feature matrices and labels for training and testing
    X_train = [list(dic_features[ent]) for ent in train_entities]
    X_test = [list(dic_features[ent]) for ent in test_entities]
    y_train = [dic_labels[ent] for ent in train_entities]
    y_test = [dic_labels[ent] for ent in test_entities]

    # For validation, use the same entities as training
    X_train_valid =  [list(dic_features[ent]) for ent in train_entities]
    X_valid =  [list(dic_features[ent]) for ent in train_entities]
    y_valid = [dic_labels[ent] for ent in train_entities]
    y_train_valid = [dic_labels[ent] for ent in train_entities]

    # Get the parameter grid for the specified algorithm
    if alg == "MLP":
        param_grid = get_parameter_grid_mlp(optimization)

    # Run the ML model for the transfer learning scenario
    metrics = {"pr":[], "rec":[], "f1":[], "acc":[], "waf":[], "auc":[]}
    metrics = run_ml_model_one_partition(param_grid, optimization, 0, path_metrics, alg, metrics, X_train, X_valid, X_train_valid, X_test, y_train, y_valid, y_train_valid, y_test)