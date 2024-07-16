import statistics
import pandas as pd
import numpy as np
import pickle
import rdflib 
from rdflib.namespace import RDF, RDFS, OWL
from scipy import stats

import pyrdf2vec
from pyrdf2vec.rdf2vec2 import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ParameterGrid

import os

from utils_datasets import process_labels_file, read_features_file, read_features_file, write_features_file
from utils_ml import write_metrics, get_parameter_grid_mlp, run_ml_model_one_partition
from utils_cv import extract_train_test_valid_entities, extract_X_train_test_valid, extract_y_train_test_valid

from multiprocessing import cpu_count
n_jobs = cpu_count()


def ensure_folder(folder_path):
    """
    Ensure that a folder exists at the specified path. If the folder does not exist, it is created.
    Parameters:
        folder_path (str): The path of the folder to check and create if it does not exist.
    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_patient_representation(path_kge_model, path_kg, patients):
    """
    Generate patient representations using a rdf2vec model.
    Parameters:
        path_kge_model (str): Path to the serialized rdf2vec model.
        path_kg (str): Path to the graph.
        patients (list): List of patient identifiers.
    Returns:
        dic_features (dict): A dictionary mapping each patient to their corresponding feature vector.
    """
    # Load the knowledge grap
    kg = KG(location=path_kg, is_remote=False, mul_req=False)

    # Load the KGE model
    with open(path_kge_model, "rb") as file_kge:
        kge_model = pickle.load(file_kge)

    # Generate walks for the specified patients
    walks = kge_model.get_walks(kg, patients)
    # Flatten the list of walks into a single corpus
    corpus = [walk for entity_walks in walks for walk in entity_walks]
    # Update the KGE model with the new corpus
    model_updated = kge_model.embedder._model
    model_updated.build_vocab(corpus, update=True)
    model_updated.train(corpus, total_examples=len(corpus), epochs=5)
    # Create a dictionary of features for each patient
    dic_features = {pat:list(model_updated.wv.get_vector(pat)) for pat in patients}          
    return dic_features


def process_representations_file(path_kge_model, path_kg, path_representations, patients):
    """
    Process patient representations by either loading existing features from a file
    or generating new features using rdf2vec model.
    Parameters:
        path_kge_model (str): Path to the serialized rdf2vec model.
        path_kg (str): Path to the graph.
        path_representations (str): Path to the output file for patient representations.
        patients (list): List of patient identifiers.
    Returns:
        dic_features (dict): A dictionary mapping each patient to their corresponding feature vector.
    """
    if os.path.exists(path_representations):
        dic_features = read_features_file(path_representations)
    
    else:
        dic_features = get_patient_representation(path_kge_model, path_kg, patients)
        write_features_file(dic_features, path_representations)
    return dic_features


def run_ml_model(path_kg, path_kge_model, path_representations, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities, path_valid_entities, optimization):
    """
    Run a ML model using patient representations.
    Parameters:
        path_kg (str): Path to the graph.
        path_kge_model (str): Path to the rdf2vec model file.
        path_representations (str): Path to the output file for patient representations.
        path_metrics (str): Path to save the metrics results.
        cv_folds (int): Number of cross-validation folds.
        alg (str): The algorithm to use (e.g., "MLP").
        path_entities_label (str): Path to the file containing entity labels.
        path_train_entities (str): Path to the training entities file.
        path_test_entities (str): Path to the testing entities file.
        path_valid_entities (str): Path to the validation entities file.
        optimization (bool): Whether to optimize hyperparameters.
    Returns:
        None
    """
    # Process labels and retrieve patient identifiers
    dic_labels = process_labels_file(path_entities_label)
    patients = [pat for pat in dic_labels]

    # Load or compute patient representations
    dic_features = process_representations_file(path_kge_model, path_kg, path_representations, patients)

    # Get hyperparameter grid for the chosen algorithm
    if alg == "MLP":
        param_grid = get_parameter_grid_mlp(optimization)

    metrics_cross_cv = {"pr":[], "rec":[], "f1":[], "acc":[], "waf":[], "auc":[]}
    for cv in range(cv_folds):
        # Extract train, validation, and test entities for the current fold
        train_entities, valid_entities, train_valid_entities, test_entities = extract_train_test_valid_entities(path_train_entities, path_valid_entities, path_test_entities, cv)
        # Extract features and labels for the current fold
        X_train, X_valid, X_train_valid, X_test = extract_X_train_test_valid(dic_features, train_entities, valid_entities, train_valid_entities, test_entities)
        y_train, y_valid, y_train_valid, y_test = extract_y_train_test_valid(dic_labels, train_entities, valid_entities, train_valid_entities, test_entities)
        # Run the ML model for the current partition
        metrics_cross_cv = run_ml_model_one_partition(param_grid, optimization, cv, path_metrics, alg, metrics_cross_cv, X_train, X_valid, X_train_valid, X_test, y_train, y_valid, y_train_valid, y_test)

    # Calculate and write average metrics
    metric_values = [statistics.mean(metrics_cross_cv["acc"]), statistics.mean(metrics_cross_cv["pr"]), statistics.mean(metrics_cross_cv["rec"]), statistics.mean(metrics_cross_cv["f1"]),statistics.mean(metrics_cross_cv["waf"]), statistics.mean(metrics_cross_cv["auc"])]
    write_metrics(path_metrics + 'AverageMetrics.tsv', ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'WAF', 'AUC'], metric_values)


def run_ml_model_transfer_learning(path_kg, path_kge_model, path_representations, path_metrics, alg, path_entities_label,  path_entities_dataset1, path_entities_dataset2, path_entities_dataset3, optimization): 
    """
    Run a ML model using transfer learning with patient representations from a knowledge graph.
    Parameters:
        path_kg (str): Path to the knowledge graph.
        path_kge_model (str): Path to the rdf2vec model file.
        path_representations (str): Path to the output file for patient representations.
        path_metrics (str): Path to save the metrics results.
        alg (str): The algorithm to use (e.g., "MLP").
        path_entities_label (str): Path to the file containing entity labels.
        path_entities_dataset1 (str): Path to the first dataset of training entities (for training).
        path_entities_dataset2 (str): Path to the second dataset of training entities (for training).
        path_entities_dataset3 (str): Path to the testing entities file (for testing).
        optimization (bool): Whether to optimize hyperparameters.
    Returns:
        None
    """
    # Process labels and retrieve patient identifiers
    dic_labels = process_labels_file(path_entities_label)
    patients = [pat for pat in dic_labels]

    # Load or compute patient representations
    dic_features = process_representations_file(path_kge_model, path_kg, path_representations, patients)
    
    # Combine entities from the first two datasets for training
    dataset1_entities = [ent.strip().split("\t")[0] for ent in open(path_entities_dataset1, 'r').readlines()]
    dataset2_entities = [ent.strip().split("\t")[0] for ent in open(path_entities_dataset2, 'r').readlines()]
    train_entities = dataset1_entities + dataset2_entities
    # Extract test entities from the third dataset
    test_entities = [ent.strip().split("\t")[0] for ent in open(path_entities_dataset3, 'r').readlines()]

    # Prepare feature matrices and labels
    X_train = [list(dic_features[ent]) for ent in train_entities]
    X_test = [list(dic_features[ent]) for ent in test_entities]
    y_train = [dic_labels[ent] for ent in train_entities]
    y_test = [dic_labels[ent] for ent in test_entities]

    # For validation, use the same training entities
    X_train_valid =  [list(dic_features[ent]) for ent in train_entities]
    X_valid =  [list(dic_features[ent]) for ent in train_entities]
    y_valid = [dic_labels[ent] for ent in train_entities]
    y_train_valid = [dic_labels[ent] for ent in train_entities]

    # Get hyperparameter grid for the chosen algorithm
    if alg == "MLP":
        param_grid = get_parameter_grid_mlp(optimization)

    # Initialize metrics and run the model
    metrics = {"pr":[], "rec":[], "f1":[], "acc":[], "waf":[], "auc":[]}
    metrics = run_ml_model_one_partition(param_grid, optimization, 0, path_metrics, alg, metrics, X_train, X_valid, X_train_valid, X_test, y_train, y_valid, y_train_valid, y_test)



