
import build_expression_data_kg
import run_prediction_kge_optimization
import run_prediction_tabulardata_optimization
import run_prediction_gnn_optimization

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


#################################### Pre-Processing data #################################### 
datasets = ["Diabetes_type_2/GSE184050",
            "Diabetes_type_2/GSE78721", 
            "Diabetes_type_2/GSE202295",
            "Coronary_artery_disease/GSE12288",
            "Coronary_artery_disease/GSE20681",
            "Coronary_artery_disease/GSE42148",
            "Breast_cancer/GSE9574",
            "Breast_cancer/GSE10810", 
            "Breast_cancer/GSE86374"]

for dataset in datasets:

    path_GO = "GO/go.owl"
    path_GO_annotations = "GO/goa_human.gaf"    
    path_gene_expression_features = dataset + "/data/Gene_expression_features.pickle"
    path_genes = dataset + "/data/Genes.tsv"

    path_output_kg =  dataset + "/data/kg_zscorelinkpatientgene.nt"
    build_expression_data_kg.build_expression_data_kg_linkpatientgene(path_gene_expression_features, path_genes, path_output_kg)
    path_output_kg_enriched =  dataset + "/data/kg_zscorelinkpatientgene_enriched_simpGO.nt"
    build_expression_data_kg.build_enriched_expression_data_simpgo_kg(path_output_kg, path_GO, path_GO_annotations, path_output_kg_enriched)


datasets = [("Breast_cancer/GSE9574_GSE10810_GSE86374/data/", "Breast_cancer/GSE9574/data/", "Breast_cancer/GSE10810/data/", "Breast_cancer/GSE86374/data/"),
            ("Coronary_artery_disease/GSE12288_GSE20681_GSE42148/data/", "Coronary_artery_disease/GSE12288/data/", "Coronary_artery_disease/GSE20681/data/", "Coronary_artery_disease/GSE42148/data/"),
            ("Diabetes_type_2/GSE184050_GSE78721_GSE202295/data/", "Diabetes_type_2/GSE78721/data/", "Diabetes_type_2/GSE184050/data/", "Diabetes_type_2/GSE202295/data/"),]

for path_output, dataset1, dataset2, dataset3 in datasets:

    ensure_folder(path_output)
    list_patient_files = [dataset1 + "Patients.tsv", dataset2 + "Patients.tsv", dataset3 + "Patients.tsv"]
    list_dic_features_file = [dataset1 + "Gene_expression_features.pickle", dataset2 + "Gene_expression_features.pickle", dataset3 + "Gene_expression_features.pickle"]
    list_gene_files = [dataset1 + "Genes.tsv", dataset2 + "Genes.tsv", dataset3 + "Genes.tsv"]
    build_expression_data_kg.unite_datasets(path_output, list_patient_files, list_dic_features_file, list_gene_files)

    path_output_kg = path_output + "kg_zscorelinkpatientgene_enriched_simpGO.nt"
    list_kg_files = [dataset1 + "/kg_zscorelinkpatientgene_enriched_simpGO.nt", 
                     dataset2 + "/kg_zscorelinkpatientgene_enriched_simpGO.nt", 
                     dataset3 + "/kg_zscorelinkpatientgene_enriched_simpGO.nt"]
    build_expression_data_kg.join_kgs(path_output_kg, list_kg_files)


#################################### Running ML methods ####################################

alg = "MLP"
emb_size = 500 
max_walks = 500
max_depth = 4
n_epochs = 500
optimization = False
if optimization:
    opt = "_opt"
else:
    opt = ""

#################################### Single-dataset learning and Multi-dataset learning #################################### 

datasets = [("Diabetes_type_2/GSE184050", 5),
            ("Diabetes_type_2/GSE78721", 5), 
            ("Diabetes_type_2/GSE202295", 5),
            ("Coronary_artery_disease/GSE12288", 5),
            ("Coronary_artery_disease/GSE20681", 5), 
            ("Coronary_artery_disease/GSE42148", 5),
            ("Breast_cancer/GSE9574", 5),
            ("Breast_cancer/GSE10810", 5), 
            ("Breast_cancer/GSE86374", 5),
            ("Diabetes_type_2/GSE184050_GSE78721_GSE202295", 15),
            ("Coronary_artery_disease/GSE12288_GSE20681_GSE42148", 15),
            ("Breast_cancer/GSE9574_GSE10810_GSE86374", 15),]

for dataset, cv_folds in datasets:

    path_entities_label = dataset + "/data/Patients.tsv" 
    path_genes_file = dataset + "/data/Genes.tsv" 
    path_features_file = dataset + "/data/Gene_expression_features.pickle"

    path_train_entities = dataset + "/Train_Entities-Train"
    path_valid_entities = dataset + "/Train_Entities-Valid"
    path_test_entities = dataset +  "/Test_Entities"

    ####### GE+MLP #######
    ensure_folder(dataset + "/GE/" + alg + opt + "/")
    path_metrics = dataset + "/GE/" + alg + opt + "/"
    run_prediction_tabulardata_optimization.run_ml_model(path_features_file, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities, path_valid_entities, optimization)
    
    type_representations = ["zscore_higher1"]
    for type_repr in type_representations:

        ######## KGE+MLP #######
        ensure_folder(dataset + "/KGE/embeddings_genes_GO_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_repr + "/" + alg + opt + "/")
        path_kge_model = "./GO/embeddings_genes_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/rdf2vec_model.pickle"
        path_representations = dataset + "/KGE/embeddings_genes_GO_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_repr + "/Patients_representations.pickle" 
        path_metrics = dataset + "/KGE/embeddings_genes_GO_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_repr + "/" + alg + opt + "/"
        path_kg = dataset + "/data/kg_zscorelinkpatientgene_enriched_simpGO.nt"
        run_prediction_kge_optimization.run_ml_model(path_kg, path_kge_model, path_representations, path_metrics, cv_folds, alg, path_entities_label, path_train_entities, path_test_entities, path_valid_entities, optimization)
        
        #### KGE+GCN #######
        ensure_folder(dataset + "/GCN" + opt + "/" + type_repr + "/")
        type_features = "embeddings"
        path_patient_representation = dataset + "/KGE/embeddings_genes_GO_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_repr + "/Patients_representations.pickle" 
        path_kge_model = "./GO/embeddings_genes_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/rdf2vec_model.pickle"
        path_kg = dataset + "/data/kg_zscorelinkpatientgene_enriched_simpGO.nt"
        path_metrics = dataset + "/GCN" + opt + "/" + type_repr +"/"
        run_prediction_gnn_optimization.run_GNN(type_repr, type_features, n_epochs, emb_size, path_kge_model, path_patient_representation, path_features_file, path_genes_file, path_kg, cv_folds, path_metrics, path_entities_label, path_train_entities, path_valid_entities, path_test_entities, optimization)

  
#################################### Transfer Learning #################################### 

datasets = [("Diabetes_type_2/GSE78721", "Diabetes_type_2/GSE202295", "Diabetes_type_2/GSE184050", "Diabetes_type_2/GSE184050-GSE78721_GSE202295", "Diabetes_type_2/GSE184050_GSE78721_GSE202295"),
            ("Diabetes_type_2/GSE202295", "Diabetes_type_2/GSE184050", "Diabetes_type_2/GSE78721", "Diabetes_type_2/GSE78721-GSE184050_GSE202295", "Diabetes_type_2/GSE184050_GSE78721_GSE202295"),
            ("Diabetes_type_2/GSE78721", "Diabetes_type_2/GSE184050", "Diabetes_type_2/GSE202295", "Diabetes_type_2/GSE202295-GSE184050_GSE78721", "Diabetes_type_2/GSE184050_GSE78721_GSE202295"),

            ("Coronary_artery_disease/GSE20681", "Coronary_artery_disease/GSE42148", "Coronary_artery_disease/GSE12288", "Coronary_artery_disease/GSE12288-GSE20681_GSE42148", "Coronary_artery_disease/GSE12288_GSE20681_GSE42148"),
            ("Coronary_artery_disease/GSE12288", "Coronary_artery_disease/GSE42148", "Coronary_artery_disease/GSE20681", "Coronary_artery_disease/GSE20681-GSE12288_GSE42148", "Coronary_artery_disease/GSE12288_GSE20681_GSE42148"),
            ("Coronary_artery_disease/GSE12288", "Coronary_artery_disease/GSE20681", "Coronary_artery_disease/GSE42148", "Coronary_artery_disease/GSE42148-GSE12288_GSE20681", "Coronary_artery_disease/GSE12288_GSE20681_GSE42148"),

            ("Breast_cancer/GSE10810", "Breast_cancer/GSE86374", "Breast_cancer/GSE9574", "Breast_cancer/GSE9574-GSE10810_GSE86374", "Breast_cancer/GSE9574_GSE10810_GSE86374"),
            ("Breast_cancer/GSE86374", "Breast_cancer/GSE9574", "Breast_cancer/GSE10810", "Breast_cancer/GSE10810-GSE9574_GSE86374", "Breast_cancer/GSE9574_GSE10810_GSE86374"),
            ("Breast_cancer/GSE9574", "Breast_cancer/GSE10810", "Breast_cancer/GSE86374", "Breast_cancer/GSE86374-GSE9574_GSE10810", "Breast_cancer/GSE9574_GSE10810_GSE86374"),]

for dataset1, dataset2, dataset3, dataset_final, dataset_feat in datasets:

    path_entities_label = dataset_feat + "/data/Patients.tsv" 
    path_genes_file = dataset_feat + "/data/Genes.tsv" 
    path_features_file = dataset_feat + "/data/Gene_expression_features.pickle"

    path_entities_dataset1 = dataset1 + "/data/Patients.tsv" 
    path_entities_dataset2 = dataset2 + "/data/Patients.tsv" 
    path_entities_dataset3 = dataset3 + "/data/Patients.tsv" 

    ###### GE+MLP #######
    ensure_folder(dataset_final + "/GE/" + alg + opt + "/")
    path_metrics = dataset_final + "/GE/" + alg + opt + "/"
    run_prediction_tabulardata_optimization.run_ml_model_transfer_learning(path_features_file, path_metrics, alg, path_entities_label,  path_entities_dataset1, path_entities_dataset2, path_entities_dataset3, optimization)

    type_representations = ["zscore_weighthedavg_higher1"]
    for type_repr in type_representations:

        ###### KGE+MLP #######
        ensure_folder(dataset_final + "/KGE/embeddings_genes_GO_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_repr + "/" + alg + opt + "/")
        path_kge_model = "./GO/embeddings_genes_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/rdf2vec_model.pickle"
        path_representations = dataset_feat + "/KGE/embeddings_genes_GO_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_repr + "/Patients_representations.pickle" 
        path_metrics = dataset_final + "/KGE/embeddings_genes_GO_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_repr + "/" + alg + opt + "/"
        path_kg = dataset_feat + "/data/kg_zscorelinkpatientgene_enriched_simpGO.nt"
        run_prediction_kge_optimization.run_ml_model_transfer_learning(path_kg, path_kge_model, path_representations, path_metrics, alg, path_entities_label,  path_entities_dataset1, path_entities_dataset2, path_entities_dataset3, optimization)
        
        ##### KGE+GCN #######
        ensure_folder(dataset_final + "/GCN" + opt + "/" + type_repr + "/")
        type_features = "embeddings"
        path_patient_representation = dataset_feat + "/KGE/embeddings_genes_GO_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/" + type_repr + "/Patients_representations.pickle" 
        path_kge_model = "./GO/embeddings_genes_" + str(emb_size) + "_" + str(max_walks) + "_" + str(max_depth) + "/rdf2vec_model.pickle"
        path_kg =  dataset_feat + "/data/kg_zscorelinkpatientgene_enriched_simpGO.nt"
        path_metrics = dataset_final + "/GCN" + opt + "/" + type_repr + "/"
        run_prediction_gnn_optimization.run_GNN_transfer_learning(type_repr, n_epochs, emb_size, path_kge_model, path_patient_representation, path_features_file, path_genes_file, path_kg, path_metrics, path_entities_label, path_entities_dataset1, path_entities_dataset2, path_entities_dataset3, optimization)
