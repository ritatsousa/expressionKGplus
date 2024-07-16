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


def read_genes_file(path_genes_file):
    """
    Read a file containing gene names and return a list of genes.
    Parameters:
        path_genes_file (str): The path to the file containing gene names.
    Returns:
        genes (list): A list of gene names.
    """
    genes = []
    with open(path_genes_file, "r") as input_gene:
        for line in input_gene:
             genes.append(line.strip())
    return genes


def read_features_file(path_features_file):
    """
    Read a file containing gene expression features and return the loaded dictionary.
    Parameters:
        path_features_file (str): The path to the file containing pickled gene expression features.
    Returns:
        dic_features (dict): A dictionary containing gene expression features.
    """
    with open(path_features_file, "rb") as features_file:
        dic_features = pickle.load(features_file)
    return dic_features


def write_genes_file(genes, path_output):
    """
        Write a list of gene names to a file.
    Parameters:
        genes (list): A list of gene names.
        path_output (str): The path to the output file where gene names will be written.
    Returns:
        None
    """
    with open(path_output, "w") as output_genes:
        for gene in genes:
            output_genes.write(gene+ "\n")


def write_features_file(dic_features, path_output):
    """
    Write a dictionary of gene expression features to a file.
    Parameters:
        dic_features (dict): A dictionary containing gene expression features.
        path_output (str): The path to the output file where the dictionary will be saved.
    Returns:
        None
    """
    with open(path_output, "wb") as features_file:
        pickle.dump(dic_features, features_file)


def process_labels_file(path_entities_label):
    """
    Read a file containing entities and their labels and return a dictionary mapping entities to labels.
    Parameters:
        path_entities_label (str): The path to the file containing entities and labels. Each line has the format: "Ent\tLabel\n"
    Returns:
        dic_labels (dict): A dictionary mapping entities to their respective labels.
    """
    dic_labels = {}
    with open(path_entities_label, 'r') as file_entities_label:
        for line in file_entities_label:
            ent, label = line.strip().split('\t')
            dic_labels[ent] = int(label)
    return dic_labels


def process_dataset(path_genes_file, path_features_file):
    """
    Process gene and feature files to create a pd.DataFrame.
    Parameters:
        path_genes_file (str): The path to the file containing gene names.
        path_features_file (str): The path to the pickled file containing gene expression features.
    Returns:
        genes, df (tuple): A tuple containing a list of gene names and a pd.DataFrame with patients as rows and genes as columns.
    """
    genes = read_genes_file(path_genes_file)
    dic_features = read_features_file(path_features_file)

    df = pd.DataFrame(dic_features)
    df = df.transpose()
    df.reset_index(inplace=True)
    df.columns = ['Patient'] + genes
    return genes, df


def transform_df_2_dic(df):
    """
    Transform a DataFrame into a dictionary of gene expression features.
    Parameters:
        df (pd.DataFrame): The DataFrame to be transformed, with 'Patient' as one of the columns.
    Returns:
        dic_features, genes (tuple): A tuple containing a dictionary of gene expression features and a list of genes.
    """
    df = df.fillna(0)
    df.set_index('Patient', inplace=True)
    df_transposed = df.transpose()
    dic_features = df_transposed.to_dict()
    for key in dic_features:
        genes = [key2 for key2 in dic_features[key]]
    dic_features = {key: list(values.values()) for key, values in dic_features.items()}
    return dic_features, genes
