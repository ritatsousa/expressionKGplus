import requests
import pickle
import csv
import statistics
from scipy import stats
import numpy as np
import pandas as pd

import rdflib
from rdflib import Graph, URIRef, XSD, Literal, Namespace, BNode, Literal, RDF

from utils_datasets import process_dataset, write_genes_file, write_features_file, transform_df_2_dic, read_genes_file

import os 
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


def build_expression_data_kg_linkpatientgene(path_gene_expression_features, path_genes, path_output_kg):

    """
    Build a knowledge graph linking patients to genes based on gene expression values.
    This function reads gene expression features and gene lists from specified files, processes the data,
    and constructs a knowledge graph. The knowledge graph is then serialized and saved to the specified output path.
    Parameters:
        path_gene_expression_features (str): Path to the file containing dictionary of gene expression features (in pickle format).
        path_genes (str): Path to the file containing the list of genes (one per line).
        path_output_kg (str): Path where the output knowledge graph will be saved.
    Returns:
        None
    """

    # Load gene expression features from the pickle file
    with open(path_gene_expression_features, 'rb') as gene_expression_features_file:
        dic_features = pickle.load(gene_expression_features_file)

    # Read genes from the file
    genes = [gene.strip() for gene in open(path_genes, 'r').readlines()]

    # Process each patient's gene expression data
    dic_expression_genes = {gene:[] for gene in genes}
    patients = []
    for patient in dic_features:
        patients.append(patient)
        for i in range(len(dic_features[patient])):
            dic_expression_genes[genes[i]].append(dic_features[patient][i])    

    # Define namespaces for the RDF graph
    ns = Namespace("http://")
    ns_patient = Namespace("http://patients/")
    ns_gene = Namespace("https://www.genecards.org/cgi-bin/carddisp.pl?gene=")
    graph = rdflib.Graph()

    # Add patient and their gene links to the graph if z-score > 1
    for gene in dic_expression_genes:
        gene_uri = ns_gene[gene]
        graph.add((gene_uri, RDF.type, ns.Gene))
        dic_expression_genes[gene] = list(stats.zscore(np.array(dic_expression_genes[gene])))
    
    for p in range(len(patients)):
        patient = patients[p].replace("http://patients/", "")
        patient_uri = ns_patient[patient]
        graph.add((patient_uri, RDF.type, ns.Patient))

        for i in range(len(genes)):
            expression_level = dic_expression_genes[genes[i]][p]
            if expression_level >= 1:
                expression_gene = ns_gene[genes[i]]
                graph.add((patient_uri, ns.hasGeneExpression, expression_gene))
  
    # Serialize the graph to the specified output file
    graph.serialize(path_output_kg, format='nt')


def build_enriched_expression_data_simpgo_kg(path_kg, path_GO, path_GO_annotations, path_output_kg_enriched):
    """
    Enrich a knowledge graph with Gene Ontology and Gene Ontology annotations.
    This function reads an existing knowledge graph, a gene ontology graph, and a file with GO annotations.
    It enriches the knowledge graph by adding GO classes and their relationships, then links genes to their
    corresponding GO annotations. The enriched knowledge graph is then serialized and saved to the specified
    output path.
    Parameters:
        path_kg (str): Path to the gene expression knowledge graph file.
        path_GO (str): Path to the Gene Ontology graph file.
        path_GO_annotations (str): Path to the file containing Gene Ontology annotations.
        path_output_kg_enriched (str): Path where the enriched knowledge graph will be saved (in N-Triples format).
    Returns:
        None
    """
     # Load the gene expression knowledge graph with links between patients and genes
    kg =  rdflib.Graph()
    kg.parse(path_kg)
    
    # Load the GO ontology graph
    kg_go = rdflib.Graph()
    kg_go.parse(path_GO)

    # Add GO classes relationships from the GO graph to the gene expression knowledge graph
    for s, p, o in kg_go.triples((None, None, None)):
        if ("http://purl.obolibrary.org/obo/GO_" in str(s)) and ("http://purl.obolibrary.org/obo/GO_" in str(o)):
            kg.add((s,p,o))

    # Define namespaces for the RDF graph
    ns = Namespace("http://")
    ns_go = Namespace("http://purl.obolibrary.org/obo/GO_")
    ns_gene = Namespace("https://www.genecards.org/cgi-bin/carddisp.pl?gene=")

    # Read GO annotations from the file
    dic_GO_annotations = {}
    with open(path_GO_annotations, 'r') as file:
        for line in file:
            if line.startswith('!'):
                continue
            fields = line.strip().split('\t')
            gene_id, go_term = fields[2], fields[4]
            if gene_id not in dic_GO_annotations:
                dic_GO_annotations[gene_id] = []
            dic_GO_annotations[gene_id].append(go_term)

    # Add gene-GO class annotations to the knowledge graph
    for gene in dic_GO_annotations:
        gene_uri = ns_gene[gene]
        kg.add((gene_uri, RDF.type, ns.gene))
        for annot in dic_GO_annotations[gene]:
            go_uri = ns_go[annot.split(":")[-1]]
            kg.add((gene_uri, ns.hasAnnotation, go_uri))

    # Serialize the enriched knowledge graph to the output file
    kg.serialize(path_output_kg_enriched, format='nt')


def unite_datasets(path_output, list_patient_files, list_dic_features_file, list_gene_files):
    """
    Unite multiple datasets into a single set of patient data and gene expression features.
    Parameters:
        path_output (str): The path where the output files will be saved.
        list_patient_files (list of str): List of file paths containing patient data.
        list_dic_features_file (list of str): List of file paths containing dictionary feature files.
        list_gene_files (list of str): List of file paths containing gene data files.
    Returns:
        None
    """

    # Combine patient data into a single file
    with open(path_output + "Patients.tsv", "w") as output_patient_file:
        for path_patient_file in list_patient_files:
            with open(path_patient_file, "r") as patient_file:
                for line in patient_file:
                    output_patient_file.write(line)
    
    genes_1, df_1 = process_dataset(list_gene_files[0], list_dic_features_file[0])
    genes_2, df_2 = process_dataset(list_gene_files[1], list_dic_features_file[1])
    genes_3, df_3 = process_dataset(list_gene_files[2], list_dic_features_file[2])
    
    common_genes_1_2 = [g for g in genes_1 if g in genes_2]
    df_merged_1_2 = pd.merge(df_1, df_2, on=['Patient']+common_genes_1_2, how='outer', suffixes=('_1', '_2'))

    common_genes_1_2_3 = [g for g in genes_3 if (g in genes_2 or g in genes_1)]
    df_merged_1_2_3 = pd.merge(df_merged_1_2, df_3, on=['Patient']+common_genes_1_2_3, how='outer', suffixes=('_1_2', '_3'))

    dic_features, genes = transform_df_2_dic(df_merged_1_2_3) 

    # Write the combined gene data to a file
    write_genes_file(genes, path_output + "Genes.tsv")
    # Write the combined dictionary features to a pickle file
    write_features_file(dic_features, path_output + "Gene_expression_features.pickle")


def join_kgs(path_output_kg, list_kg_files):
    """
    Join multiple RDF knowledge graphs into a single knowledge graph.
    Parameters:
        path_output_kg (str): The path where the combined knowledge graph will be saved (in N-Triples format).
        list_kg_files (list of str): List of file paths containing the RDF knowledge graphs to be combined.
    Returns:
        None
    """
    kg =  rdflib.Graph()
    for kg_file in list_kg_files:
        kg.parse(kg_file)
    kg.serialize(path_output_kg, format='nt')


