import torch
from torch_geometric.data import Data
import json

def process_ast(ast):
    # Convert AST to PyTorch Geometric format
    node_features = torch.tensor([node['type'] for node in ast['nodes']], dtype=torch.float)
    edge_index = []
    for src, dsts in ast['adjacencyList'].items():
        for dst in dsts:
            edge_index.append([int(src), dst])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    labels = torch.tensor([node['nullable'] for node in ast['nodes']], dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, y=labels)

# Assuming 'asts' is a list of ASTs in the given format
asts = [json.loads(ast) for ast in asts_json]
graphs = [process_ast(ast) for ast in asts]
