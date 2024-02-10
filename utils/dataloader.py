"""
Planetoid is a citation network dataset from Cora, CiteSeer, and PubMed. The nodes are documents with 1433-dimensional 
bag-of-words feature vectors, and the edges are citation links between research papers. 
There are 7 classes, and we will train the model to predict missing labels.
We will ingest the Planetoid Cora dataset, and row normalize the bag of words input features. 
After that, we will analyze the dataset.

The Cora dataset has 2708 nodes, 10,556 edges, 1433 features, and 7 classes. 
The first object has 2708 train, validation, and test masks. 
We will use these masks to train and evaluate the model.
"""

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def planetoid_dataset(normalize_features=True):

    # Check if we will apply feature normalization
    if normalize_features:
        transform= NormalizeFeatures()
    else:
        transform= None

    # Load the dataset
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=transform)

    # Splitting the dataset
    dataset.train_mask.fill_(False)
    dataset.train_mask[:dataset[0].num_nodes - 1000] = 1
    dataset.val_mask.fill_(False)
    dataset.val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
    dataset.test_mask.fill_(False)
    dataset.test_mask[dataset[0].num_nodes - 500:] = 1 

    print(f"dataset: {dataset}")
    print('=====================')
    print(f"Number of nodes: {dataset[0].num_nodes}")
    print(f"Number of features: {dataset.num_node_features}")
    print(f"Number of edges: {dataset[0].num_edges}")
    print(f"Number of classes: {dataset.num_classes}")
        
    return dataset