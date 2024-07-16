import statistics
import os
import rdflib
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
from sklearn import metrics
import numpy as np
from scipy import stats
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import pyrdf2vec

from utils_datasets import process_labels_file
from utils_ml import write_metrics, compute_metrics, write_predictions, store_metrics
from utils_cv import extract_train_test_valid_entities

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU available")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")


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


def get_parameter_grid_gcn(optimization, in_channels, out_channels):
    """
    Generate a hyperparameter grid for a GCN.
    Parameters:
        optimization (bool): Indicates whether to optimize hyperparameters or use default values.
        in_channels (int): Number of input channels for the GCN.
        out_channels (int): Number of output channels for the GCN.
    Returns:
        param_grid (dict): A dictionary containing the hyperparameter grid.
    """
    if optimization:
        param_grid = {'hidden_channels': [16, 32, 64], 
                    'n_conv': [2,3,4,5,6],
                    'lr': [0.01, 0.1, 0.2, 0.001],
                    'dropout': [0, 0.1, 0.2, 0.3, 0.5],
                    'in_channels': [in_channels],
                    'out_channels': [out_channels],
                    'aggr': ['add', 'mean', 'max']
                    }
    else:
        param_grid = {'hidden_channels': [16], 
                    'n_conv': [2],
                    'lr': [0.001],
                    'dropout': [0.05],
                    'in_channels': [in_channels],
                    'out_channels': [out_channels],
                    'aggr': ['mean']
                    }
    return param_grid


class GCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_conv, dropout, aggr):
        self.n_conv=n_conv
        self.dropout = dropout
        super(GCNN, self).__init__()

        # Define convolutional layers
        self.conv1 = GCNConv(in_channels, hidden_channels, aggr=aggr)

        if self.n_conv==2:
            self.conv2 = GCNConv(hidden_channels, out_channels, aggr=aggr)
        
        elif self.n_conv==3:
            self.conv2 = GCNConv(hidden_channels, hidden_channels, aggr=aggr)
            self.conv3 = GCNConv(hidden_channels, out_channels, aggr=aggr)

        elif self.n_conv==4:
            self.conv2 = GCNConv(hidden_channels, hidden_channels, aggr=aggr)
            self.conv3 = GCNConv(hidden_channels, hidden_channels, aggr=aggr)
            self.conv4 = GCNConv(hidden_channels, out_channels, aggr=aggr)

        elif self.n_conv==5:
            self.conv2 = GCNConv(hidden_channels, hidden_channels, aggr=aggr)
            self.conv3 = GCNConv(hidden_channels, hidden_channels, aggr=aggr)
            self.conv4 = GCNConv(hidden_channels, hidden_channels, aggr=aggr)
            self.conv5 = GCNConv(hidden_channels, out_channels, aggr=aggr)

        elif self.n_conv==6:
            self.conv2 = GCNConv(hidden_channels, hidden_channels, aggr=aggr)
            self.conv3 = GCNConv(hidden_channels, hidden_channels, aggr=aggr)
            self.conv4 = GCNConv(hidden_channels, hidden_channels, aggr=aggr)
            self.conv5 = GCNConv(hidden_channels, hidden_channels, aggr=aggr)
            self.conv6 = GCNConv(hidden_channels, out_channels, aggr=aggr)
        
        # self.out = nn.Linear(out_channels, 2)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=self.dropout)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        if self.n_conv==2:
            x = self.conv2(x, edge_index, edge_weight)
            
        elif self.n_conv==3:
            x = self.conv2(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv3(x, edge_index, edge_weight)

        elif self.n_conv==4:
            x = self.conv2(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv3(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv4(x, edge_index, edge_weight)

        elif self.n_conv==5:
            x = self.conv2(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv3(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv4(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv5(x, edge_index, edge_weight)

        elif self.n_conv==6:
            x = self.conv2(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv3(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv4(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv5(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv6(x, edge_index, edge_weight)
        
        # x =self.out(x)
        return x


def train_model(graph_data, model, n_epochs, lr):
    """
    Trains a GCN model.
    Parameters:
        graph_data (Data): Data object containing node features (x), edge indices (edge_index), edge attributes (edge_attr), training mask (train_mask), and labels (y).
        model (GCNN): GCN model instance to be trained.
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
    Returns:
        model (GCNN): Trained graph neural network model.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        # Compute loss
        loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        loss.backward()
        optimizer.step()
        # Print loss every 10 epochs
        if epoch%10 ==0:
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}')

    return model


def train_model_final(graph_data, model, n_epochs, path_folder_output, lr):
    """
    Trains a GCN model and saves loss values and visualizations.
    Parameters:
        graph_data (Data): Data object containing node features (x), edge indices (edge_index), edge attributes (edge_attr), training mask (train_mask), validation mask (train_valid_mask), and labels (y).
        model (GCNN): GCN instance to be trained.
        n_epochs (int): Number of training epochs.
        path_folder_output (str): Path to the output folder where loss values and visualizations are saved.
        lr (float): Learning rate for the optimizer.
    Returns:
        model (GCNN): Trained GCN model.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Open a file to log loss values and F1 scores
    file_loss = open(path_folder_output + "_LossValues.tsv", "w")
    file_loss.write("Epoch\tLoss\tF1\n")

    # Train the model
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        # Visualize initial embeddings for the first epoch
        if epoch == 0:
            visualize(out[graph_data.train_valid_mask], color=graph_data.y[graph_data.train_valid_mask],path_output=path_folder_output+"_Plot_initial_embeddings_training.png")
            visualize(out[graph_data.test_mask], color=graph_data.y[graph_data.test_mask],path_output=path_folder_output+"_Plot_initial_embeddings_test.png")
        # Compute loss
        loss = criterion(out[graph_data.train_valid_mask], graph_data.y[graph_data.train_valid_mask])
        loss.backward()
        optimizer.step()
        # Calculate F1 score
        _, predictions = out[graph_data.train_mask].max(dim=1)
        f1 = metrics.f1_score(graph_data.y[graph_data.train_mask].detach().numpy(), predictions.numpy())
        file_loss.write(str(epoch) + "\t" + str(loss.item()) + "\t" + str(f1) + "\n")
        # Print loss and F1 score every 10 epochs
        if epoch%10 ==0:
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}, F1: {f1}')

    # Visualize final embeddings
    visualize(out[graph_data.train_valid_mask],color=graph_data.y[graph_data.train_valid_mask],path_output=path_folder_output+"_Plot_final_embeddings_training.png")
    visualize(out[graph_data.test_mask],color=graph_data.y[graph_data.test_mask],path_output=path_folder_output+"_Plot_final_embeddings_test.png")

    file_loss.close()
    return model


def buildIds(g, dic_weights, dic_features, emb_size):
    """
    Constructs a graph representation from a given list of edges, including node IDs, edge weights, and node features.
    Parameters:
        g (iterable): An iterable of tuples representing the edges of the graph, where each tuple contains (subject, predicate, object).
        dic_weights (dict): A dictionary mapping edges (as tuples) to their corresponding weights.
        dic_features (dict): A dictionary mapping node identifiers to their feature vectors.
        emb_size (int): The size of the embedding vector for each node.
    Returns:
        edges (list): A list of tuples representing the edges in the graph, where each tuple contains (node_id_subject, node_id_object).
        edge_weights (list): A list of weights corresponding to the edges.
        node_features (torch.Tensor): A tensor containing the feature vectors for each node in the graph.
        dic_nodes (dict): A dictionary mapping node identifiers to their corresponding unique IDs.
    """
    dic_nodes = {}
    id_node = 0
    edges, edge_weights, node_features = [], [], []

    for (subj, pred, obj) in g:
        # Add subject node
        if str(subj) not in dic_nodes:
            dic_nodes[str(subj)] = id_node
            if str(subj) in dic_features:
                node_features.append(dic_features[str(subj)])
            else:
                node_features.append([0 for i in range(emb_size)])
            id_node = id_node + 1
        # Add object node
        if str(obj) not in dic_nodes:
            dic_nodes[str(obj)] = id_node
            if str(obj) in dic_features:
                node_features.append(dic_features[str(obj)])
            else:
                node_features.append([0 for i in range(emb_size)])
            id_node = id_node + 1
        # Get edge weight
        if (str(subj), str(obj)) in dic_weights:
            edge_weights.append(dic_weights[(str(subj), str(obj))])
        else:
            edge_weights.append(1)
        # Add edge
        edges.append((dic_nodes[str(subj)], dic_nodes[str(obj)]))

    return edges, edge_weights, torch.tensor(node_features), dic_nodes


def generate_weights_gene_expression(type_repr, path_gene_expression_features, path_genes, path_patient_representation, path_kge_model):
    """
    Generates weights for gene expressions for patients and retrieves node features from a KGE model.
    Parameters:
        type_repr (str): The type of representation to use.
        path_gene_expression_features (str): The file path to the gene expression features for patients.
        path_genes (str): The file path containing a list of gene identifiers.
        path_patient_representation (str): The file path to the patient representation features.
        path_kge_model (str): The file path to the rdf2vec model.
    Returns:
        dic_weights (dict): A dictionary mapping (patient, gene) tuples to their corresponding weights.
        dic_node_features (dict): A dictionary mapping node identifiers to their feature vectors.
    """
    # Load gene expression features
    dic_node_features = {}
    with open(path_gene_expression_features, 'rb') as gene_expression_features_file:
        dic_gene_expression = pickle.load(gene_expression_features_file)
    # Load genes
    genes = [gene.strip() for gene in open(path_genes, 'r').readlines()]

    # Load KGE model and retrieve node features
    with open(path_kge_model, "rb") as file_kge:
        kge_model = pickle.load(file_kge)
        vocabulary = kge_model.embedder._model.wv.key_to_index
        for ent in list(vocabulary.keys()):
            dic_node_features[str(ent)] = list(kge_model.embedder.transform([ent]))[0]

    # Generate weights for each patient's gene expression
    dic_weights = {}
    for patient in dic_gene_expression:

        values_normalized = list(stats.zscore(np.array(dic_gene_expression[patient])))
        values_normalized = [0 if x < 1 else float(x) for x in values_normalized]

        for i in range(len(dic_gene_expression[patient])):

            if (patient, "https://www.genecards.org/cgi-bin/carddisp.pl?gene=" + genes[i]) in dic_weights:
                if  dic_weights[(patient, "https://www.genecards.org/cgi-bin/carddisp.pl?gene=" + genes[i])] == 0:
                    dic_weights[(patient, "https://www.genecards.org/cgi-bin/carddisp.pl?gene=" + genes[i])] = values_normalized[i]
            else:
                dic_weights[(patient, "https://www.genecards.org/cgi-bin/carddisp.pl?gene=" + genes[i])] = values_normalized[i]

    # Load patient representations and add them to node features
    with open(path_patient_representation, "rb") as features_file:
        dic_patient_representations = pickle.load(features_file)
        for pat in dic_patient_representations:
            dic_node_features[pat] = dic_patient_representations[pat]

    return dic_weights, dic_node_features


def run_GNN_one_partition(graph_data, train_nodes, valid_nodes, train_valid_nodes, test_nodes, train_labels, valid_labels, train_valid_labels, test_labels, param_grid, optimization, in_channels, out_channels, n_epochs, path_metrics, cv, metrics_cross_cv):
    """
    Trains and evaluates a GCN on a single partition of the dataset.
    Parameters:
        graph_data (Data): The graph data containing node features, edge indices, and labels.
        train_nodes (list): List of node indices for the training set.
        valid_nodes (list): List of node indices for the validation set.
        train_valid_nodes (list): List of node indices for the combined training and validation set.
        test_nodes (list): List of node indices for the test set.
        train_labels (Tensor): Tensor containing the labels for the training set.
        valid_labels (Tensor): Tensor containing the labels for the validation set.
        train_valid_labels (Tensor): Tensor containing the labels for the combined training and validation set.
        test_labels (Tensor): Tensor containing the labels for the test set.
        param_grid (dict): Dictionary containing hyperparameter options for model optimization.
        optimization (bool): Whether to optimize hyperparameters.
        in_channels (int): Number of input features for the model.
        out_channels (int): Number of output classes for the model.
        n_epochs (int): Number of training epochs.
        path_metrics (str): Directory path to save metrics and results.
        cv (int): Cross-validation fold number.
        metrics_cross_cv (dict): Dictionary to store metrics across different cross-validation folds.
    Returns:
        metrics_cross_cv (dict): Updated dictionary containing metrics for the current partition.
    """
    # Create train and test datasets
    graph_data.y = torch.full((graph_data.num_nodes,), -1, dtype=torch.long)
    graph_data.y[train_nodes] = train_labels
    graph_data.y[valid_nodes] = valid_labels
    graph_data.y[train_valid_nodes] = train_valid_labels
    graph_data.y[test_nodes] = test_labels
    graph_data.train_mask = train_nodes
    graph_data.valid_mask = valid_nodes
    graph_data.train_valid_mask = train_valid_nodes
    graph_data.test_mask = test_nodes

    graph_data.x = graph_data.x.float()
    graph_data.x = graph_data.x.to(device)
    graph_data.edge_index = graph_data.edge_index.to(device)
    graph_data.y = graph_data.y.to(device)
        
    param_combinations = ParameterGrid(param_grid)
    best_waf = 0
    best_params = {}
        
    if optimization:
        # Hyperparameter optimization loop
        for params in param_combinations:
            hidden_channels, n_conv, lr, dropout, aggr = params['hidden_channels'], params['n_conv'], params['lr'], params['dropout'], params['aggr']
            model = GCNN(in_channels, hidden_channels, out_channels, n_conv, dropout, aggr)
            trained_model = train_model(graph_data, model, n_epochs, lr)
                
            output = trained_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            _, predictions = output[valid_nodes].max(dim=1)
            current_waf = metrics.f1_score(graph_data.y[valid_nodes].numpy(), predictions.numpy(), average="weighted")

            if current_waf > best_waf:
                best_waf = current_waf
                best_params = params
    else:
        for params in param_combinations:
            best_params = params

    # Save the best parameters to a file
    with open(path_metrics + str(cv) + '_Best_parameters.txt', 'w') as file_parameters:
        file_parameters.write(str(best_params))

    # Train final model using the best parameters
    hidden_channels, n_conv, lr, dropout, aggr = best_params['hidden_channels'], best_params['n_conv'], best_params['lr'], best_params['dropout'], best_params['aggr']
    final_model = GCNN(in_channels, hidden_channels, out_channels, n_conv, dropout, aggr)
    final_model = train_model_final(graph_data, final_model, n_epochs, path_metrics + str(cv), lr)
    
    # Make predictions on test and train sets
    output = final_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    _, predictions = output[test_nodes].max(dim=1)
    _, predictions_train = output[train_nodes].max(dim=1)

    # Compute metrics for test and training sets
    pr, rec, f1, acc, waf = compute_metrics(graph_data.y[test_nodes].numpy(), predictions.numpy())
    pr_train, rec_train, f1_train, acc_train, waf_train = compute_metrics(graph_data.y[train_nodes].numpy(), predictions_train.numpy())

    # Write predictions and metrics to files
    write_predictions(path_metrics + str(cv) + '_PredictionsTrain.tsv', graph_data.y[train_nodes], predictions_train)
    write_predictions(path_metrics + str(cv) + '_PredictionsTest.tsv', graph_data.y[test_nodes], predictions)
    write_metrics(path_metrics + str(cv) + '_MetricsTest.tsv', ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'WAF'], [acc, pr, rec, f1, waf]) 
    write_metrics(path_metrics + str(cv) + '_MetricsTrain.tsv', ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'WAF'], [acc_train, pr_train, rec_train, f1_train, waf_train])

    return store_metrics(metrics_cross_cv, ["acc", "pr", "rec", "f1", "waf"], [acc, pr, rec, f1, waf])


def run_GNN(type_repr, type_features, n_epochs, emb_size, path_kge_model, path_patient_representation, path_gene_expression_features, path_genes, path_kg, cv_folds, path_metrics, path_entities_label, path_train_entities, path_valid_entities, path_test_entities, optimization):
    """
    Executes the GCN training and evaluation process.
    Parameters:
        type_repr (str): Type of representation to be used.
        type_features (str): Type of features for the graph nodes.
        n_epochs (int): Number of training epochs for the GNN.
        emb_size (int): Size of the node embeddings.
        path_kge_model (str): Path to the rdf2vec embedding model.
        path_patient_representation (str): Path to the patient representations.
        path_gene_expression_features (str): Path to gene expression features.
        path_genes (str): Path to the list of genes.
        path_kg (str): Path to the graph file.
        cv_folds (int): Number of cross-validation folds.
        path_metrics (str): Path to save metrics and results.
        path_entities_label (str): Path to the file containing entity labels.
        path_train_entities (str): Path to the training entities file.
        path_valid_entities (str): Path to the validation entities file.
        path_test_entities (str): Path to the test entities file.
        optimization (bool): Whether to optimize hyperparameters.
    Returns:
        None
    """
    # Load entity labels
    dic_labels = process_labels_file(path_entities_label)

    # Create and parse the RDF graph from the knowledge graph file
    g = rdflib.Graph()
    g.parse(path_kg) ## , format='xml')

    # Generate weights and node features from gene expression data
    dic_weights, dic_features = generate_weights_gene_expression(type_repr, path_gene_expression_features, path_genes, path_patient_representation, path_kge_model)
    # Build edges, edge weights, node features, and node dictionary from the graph
    edges, edge_weights, node_features, dic_nodes = buildIds(g, dic_weights, dic_features, emb_size)
    # Construct the graph data structure for PyTorch Geometric
    graph_data = Data(x=torch.tensor(node_features), edge_index=torch.tensor(edges).t().contiguous(), edge_attr=torch.tensor(edge_weights))
    
    in_channels = node_features.size(1)
    out_channels = 2 
    param_grid = get_parameter_grid_gcn(optimization, in_channels, out_channels)
   
    # Loop over each fold for cross-validation
    metrics_cross_cv = {"pr":[], "rec":[], "f1":[], "acc":[], "waf":[], "auc":[]}
    for cv in range(cv_folds):
        # Extract train, validation, and test entities for the current fold
        train_entities, valid_entities, train_valid_entities, test_entities = extract_train_test_valid_entities(path_train_entities, path_valid_entities, path_test_entities, cv)

        # Convert entity to tensors
        train_nodes = torch.tensor([dic_nodes[ent] for ent in train_entities])
        valid_nodes = torch.tensor([dic_nodes[ent] for ent in valid_entities])
        train_valid_nodes = torch.tensor([dic_nodes[ent] for ent in train_valid_entities])
        test_nodes = torch.tensor([dic_nodes[ent] for ent in test_entities])

        # Convert entity labels to tensors
        train_labels = torch.tensor([dic_labels[ent] for ent in train_entities], dtype=torch.long)
        valid_labels = torch.tensor([dic_labels[ent] for ent in valid_entities], dtype=torch.long)
        train_valid_labels = torch.tensor([dic_labels[ent] for ent in train_valid_entities], dtype=torch.long)
        test_labels = torch.tensor([dic_labels[ent] for ent in test_entities], dtype=torch.long)

        # Run GNN training and evaluation for the current partition
        metrics_cross_cv = run_GNN_one_partition(graph_data, train_nodes, valid_nodes, train_valid_nodes, test_nodes, train_labels, valid_labels, train_valid_labels, test_labels, param_grid, optimization, in_channels, out_channels, n_epochs, path_metrics, cv, metrics_cross_cv)

    # Calculate and write average metrics across all folds
    metric_values = [statistics.mean(metrics_cross_cv["acc"]), statistics.mean(metrics_cross_cv["pr"]), statistics.mean(metrics_cross_cv["rec"]), statistics.mean(metrics_cross_cv["f1"]),statistics.mean(metrics_cross_cv["waf"])]
    write_metrics(path_metrics + 'AverageMetrics.tsv', ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'WAF'], metric_values)


def run_GNN_transfer_learning(type_repr, n_epochs, emb_size, path_kge_model, path_patient_representation, path_gene_expression_features, path_genes, path_kg, path_metrics, path_entities_label, path_entities_dataset1, path_entities_dataset2, path_entities_dataset3, optimization):
    """
    Executes the transfer learning process using a GCN.
    Parameters:
        type_repr (str): Type of representation to be used.
        n_epochs (int): Number of training epochs for the GNN.
        emb_size (int): Size of the node embeddings.
        path_kge_model (str): Path to the rdf2vec embedding model.
        path_patient_representation (str): Path to the patient representations.
        path_gene_expression_features (str): Path to gene expression features.
        path_genes (str): Path to the list of genes.
        path_kg (str): Path to the knowledge graph file.
        path_metrics (str): Path to save metrics and results.
        path_entities_label (str): Path to the file containing entity labels.
        path_entities_dataset1 (str): Path to the first dataset of entities for training.
        path_entities_dataset2 (str): Path to the second dataset of entities for training.
        path_entities_dataset3 (str): Path to the dataset of entities for testing.
        optimization (bool): Whether to optimize hyperparameters.
    Returns:
        None
    """
    # Load entity labels
    dic_labels = process_labels_file(path_entities_label)

    # Create and parse the RDF graph from the knowledge graph file
    g = rdflib.Graph()
    g.parse(path_kg) ## , format='xml')
    
    # Generate weights and node features from gene expression data
    dic_weights, dic_features = generate_weights_gene_expression(type_repr, path_gene_expression_features, path_genes, path_patient_representation, path_kge_model)
    # Build edges, edge weights, node features, and node dictionary from the graph
    edges, edge_weights, node_features, dic_nodes = buildIds(g, dic_weights, dic_features, emb_size)
    # Construct the graph data structure for PyTorch Geometric
    graph_data = Data(x=torch.tensor(node_features), edge_index=torch.tensor(edges).t().contiguous(), edge_attr=torch.tensor(edge_weights))
    
    in_channels = node_features.size(1)
    out_channels = 2  # Number of classes for prediction
    param_grid = get_parameter_grid_gcn(optimization, in_channels, out_channels)

    # Load entities from the training datasets
    dataset1_entities = [ent.strip().split("\t")[0] for ent in open(path_entities_dataset1, 'r').readlines()]
    dataset2_entities = [ent.strip().split("\t")[0] for ent in open(path_entities_dataset2, 'r').readlines()]
    # Combine training entities from both datasets
    train_entities = dataset1_entities + dataset2_entities

    # Load entities from the testing dataset
    test_entities = [ent.strip().split("\t")[0] for ent in open(path_entities_dataset3, 'r').readlines()]

    # Convert entity to tensors
    train_nodes = torch.tensor([dic_nodes[ent] for ent in train_entities])
    test_nodes = torch.tensor([dic_nodes[ent] for ent in test_entities])

    # Convert entity labels to tensors
    train_labels = torch.tensor([dic_labels[ent] for ent in train_entities], dtype=torch.long)
    test_labels = torch.tensor([dic_labels[ent] for ent in test_entities], dtype=torch.long)

    valid_nodes =  torch.tensor([dic_nodes[ent] for ent in train_entities])
    train_valid_nodes =  torch.tensor([dic_nodes[ent] for ent in train_entities])
    valid_labels = torch.tensor([dic_labels[ent] for ent in train_entities], dtype=torch.long)
    train_valid_labels = torch.tensor([dic_labels[ent] for ent in train_entities], dtype=torch.long)

    # Run GNN training and evaluation
    metrics = {"pr":[], "rec":[], "f1":[], "acc":[], "waf":[]}
    metrics_cross_cv = run_GNN_one_partition(graph_data, train_nodes, valid_nodes, train_valid_nodes, test_nodes, train_labels, valid_labels, train_valid_labels, test_labels, param_grid, optimization, in_channels, out_channels, n_epochs, path_metrics, 0, metrics)


def visualize(h, color, path_output):
    """
    Visualizes the embeddings in a 2D scatter plot.
    Parameters:
        h (torch.Tensor): The embeddings to visualize, expected to be of shape (n_samples, 2).
        color (torch.Tensor): The labels for each point in the scatter plot.
        path_output (str): The path where the output plot will be saved.
    Returns:
    - None
    """
    # Create a new figure for the plot
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    # Create a custom colormap ranging from red to green
    custom_cmap = LinearSegmentedColormap.from_list("red_green", ['red', 'green'])

    # Create a scatter plot of the embeddings
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], c=color, cmap=custom_cmap)
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)])
    
    # Save the plot to the specified output path
    plt.savefig(path_output)