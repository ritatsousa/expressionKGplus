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
from sklearn.model_selection import train_test_split
import utils_datasets

def extract_train_test_valid_entities(path_train_entities, path_valid_entities, path_test_entities, cv):
    """
    Extract entities for training, validation, and testing from files for a given cross-validation fold.
    Parameters:
        path_train_entities (str): The base path to the training entities file (without the '_cv.tsv' suffix).
        path_valid_entities (str): The base path to the validation entities file (without the '_cv.tsv' suffix).
        path_test_entities (str): The base path to the testing entities file (without the '_cv.tsv' suffix).
        cv (int): The cross-validation fold number.
    Returns:
        train_entities, valid_entities, train_valid_entities, test_entities (tuple): A tuple containing four lists - training entities, validation entities, combined training and validation entities, and testing entities.
    """
    train_entities = [ent.strip() for ent in open(path_train_entities + '_' + str(cv) + '.tsv', 'r').readlines()]
    valid_entities = [ent.strip() for ent in open(path_valid_entities + '_' + str(cv) + '.tsv', 'r').readlines()]
    train_valid_entities = train_entities + valid_entities
    test_entities = [ent.strip() for ent in open(path_test_entities + '_' + str(cv) + '.tsv', 'r').readlines()]
    return train_entities, valid_entities, train_valid_entities, test_entities


def extract_X_train_test_valid(dic_features, train_entities, valid_entities, train_valid_entities, test_entities):
    """
    Extract feature vectors for training, validation, combined training and validation, and testing entities.
    Parameters:
        dic_features (dict): A dictionary where keys are entities and values are lists of features.
        train_entities (list): A list of training entities.
        valid_entities (list): A list of validation entities.
        train_valid_entities (list): A list of combined training and validation entities.
        test_entities (list): A list of testing entities.
    Returns:
        X_train, X_valid, X_train_valid, X_test (tuple): A tuple containing four lists of feature vectors:
           - Feature vectors for training entities.
           - Feature vectors for validation entities.
           - Feature vectors for combined training and validation entities.
           - Feature vectors for testing entities.
    """
    X_train = [list(dic_features[ent]) for ent in train_entities]
    X_valid = [list(dic_features[ent]) for ent in valid_entities]
    X_train_valid = [list(dic_features[ent]) for ent in train_valid_entities] 
    X_test = [list(dic_features[ent]) for ent in test_entities]
    return X_train, X_valid, X_train_valid, X_test


def extract_y_train_test_valid(dic_labels, train_entities, valid_entities, train_valid_entities, test_entities):
    """
    Extract labels for training, validation, combined training and validation, and testing entities.
    Parameters:
        dic_labels (dict): A dictionary where keys are entities and values are their corresponding labels.
        train_entities (list): A list of training entities.
        valid_entities (list): A list of validation entities.
        train_valid_entities (list): A list of combined training and validation entities.
        test_entities (list): A list of testing entities.
    Returns:
        y_train, y_valid, y_train_valid, y_test (tuple): A tuple containing four lists of labels:
           - Labels for training entities.
           - Labels for validation entities.
           - Labels for combined training and validation entities.
           - Labels for testing entities.
    """
    y_train = [dic_labels[ent] for ent in train_entities]
    y_valid = [dic_labels[ent] for ent in valid_entities]
    y_train_valid = [dic_labels[ent] for ent in train_valid_entities]
    y_test = [dic_labels[ent] for ent in test_entities]
    return y_train, y_valid, y_train_valid, y_test


def split_dataset(path_entities_label, path_train_entities, path_test_entities, cv_folds, shuffle=False):
    """
    Split entities into training and testing sets using Stratified K-Fold cross-validation.
    Parameters:
        path_entities_label (str): The path to the file containing entity labels.
        path_train_entities (str): The base path for output training entities files (will append '_cv.tsv').
        path_test_entities (str): The base path for output testing entities files (will append '_cv.tsv').
        cv_folds (int): The number of cross-validation folds to create.
        shuffle (bool): Whether to shuffle the data before splitting (default is False).
    Returns:
        None
    """
    # Process labels from the provided file into a dictionary
    dic_labels = utils_datasets.process_labels_file(path_entities_label)
    entities = [key for key in dic_labels]
    # Initialize Stratified K-Fold cross-validator
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle)
    # Perform the splitting for each cross-validation fold
    for cv, (train_index, test_index) in enumerate(skf.split(entities, [dic_labels[ent] for ent in entities])):
        # Write training entities to file
        with open(path_train_entities + '_' + str(cv) + '.tsv', 'w') as train_entities_file:
            for ind in train_index:
                train_entities_file.write(entities[ind] + '\n')
        # Write testing entities to file
        with open(path_test_entities + '_' + str(cv) + '.tsv', 'w') as test_entities_file:
            for ind in test_index:
                test_entities_file.write(entities[ind] + '\n')


def split_training_set(cv_folds, path_entities_label, path_train_entities, path_train_output, path_valid_output):
    """
    Split training entities into training and validation sets for each cross-validation fold.
    Parameters:
        cv_folds (int): The number of cross-validation folds to process.
        path_entities_label (str): The path to the file containing entity labels.
        path_train_entities (str): The base path for input training entities files (will append '_cv.tsv').
        path_train_output (str): The base path for output training sets files (will append '_cv.tsv').
        path_valid_output (str): The base path for output validation sets files (will append '_cv.tsv').
    Returns:
        None
    """
    dic_labels = utils_datasets.process_labels_file(path_entities_label)
    entities = [key for key in dic_labels]
    for cv in range(cv_folds):
        train_entities = [ent.strip() for ent in open(path_train_entities + '_' + str(cv) + '.tsv', 'r').readlines()]
        labels = [dic_labels[ent] for ent in  train_entities]
        X_train, X_test, y_train, y_test = train_test_split( train_entities, labels, test_size=0.1, stratify=labels)
        with open(path_train_output + '_' + str(cv) + '.tsv', 'w') as train_output:
            for ind in X_train:
                train_output.write(ind + '\n')
        with open(path_valid_output + '_' + str(cv) + '.tsv', 'w') as valid_output:
            for ind in X_test:
                valid_output.write(ind + '\n')


def merge_splits(list_split_files, path_output):
    """
    Merge split files into training and test entity files for cross-validation.
    Parameters:
        list_split_files (list): A list of file paths for the split entity files.
        path_output (str): The directory path where the output files will be saved.
    Returns:
        None
    """
    cv = 0
    for split_file in list_split_files:
        with open(path_output + "Test_Entities_" + str(cv) + ".tsv", "w") as test_file:
            with open(split_file, "r") as input:
                for line in input:
                    test_file.write(line)
        with open(path_output + "Train_Entities_" + str(cv) + ".tsv", "w") as train_file:
            for split_file2 in list_split_files:
                if split_file2 != split_file:
                    with open(split_file2, "r") as input:
                        for line in input:
                            train_file.write(line)
        cv = cv + 1


def merge_splits_with_validation(list_split_files, path_output):
    """
    Merge split files into training and validation entity files for cross-validation.
    Parameters:
        list_split_files (list): A list of file paths for the split entity files.
        path_output (str): The directory path where the output files will be saved.
    Returns:
        None
    """
    cv = 0
    for split_file in list_split_files:
        with open(path_output + "Train_Entities-Valid_" + str(cv) + ".tsv", "w") as test_file:
            valid_entities = []
            with open(split_file.replace("Test_Entities", "Train_Entities-Valid"), "r") as input:
                for line in input:
                    valid_entities.append(line.strip())
                    test_file.write(line)
        with open(path_output + "Train_Entities-Train_" + str(cv) + ".tsv", "w") as train_file:
            with open(path_output + "Train_Entities_" + str(cv) + ".tsv", "r") as input:
                for line in input:
                    if line.strip() not in valid_entities:
                        train_file.write(line)
        cv = cv + 1