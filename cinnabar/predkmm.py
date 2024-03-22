from sklearn.cluster import KMeans
import pickle
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
import json

def getobj(file):
   while True:
    s=file.read(1)
    if not s:
        return s
    if s=='{':
       break
   depth=1
   while depth>0:
      char=file.read(1)
      if char=='{':
         depth+=1
      if char=='}':
         depth-=1
      s+=char
   return s

max_size=1195
modDir="/home/k/ks225/nullproj/cinnabar/"

def pad_adjacency_matrices(adjacency_matrices):
    global max_size

    # Pad each matrix to the maximum size
    padded_matrices = []
    for mat in adjacency_matrices:
        pad_size = max_size - mat.shape[0]
        padded_mat = np.pad(mat, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=-1)
        padded_matrices.append(padded_mat)

    return padded_matrices

# Load the model from the file
with open(modDir+'kmmodel.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Now you can use `loaded_model` as you would normally use a fitted KMeans object
# For example, you can predict the cluster for new data

with open(modDir+"temp_output.json", "r") as file:
    obj_str = getobj(file)
    #if not obj_str:
    #    break
    json_obj = json.loads(obj_str)

    nodes=json_obj['nodes']
    alist=json_obj['adjacencyList']
    amat=np.zeros((len(nodes), len(nodes)))
    for node in nodes:
        if str(node['id']) in alist:
            for neighbor in alist[str(node['id'])]:
                amat[node['id']-1,neighbor-1]=1

    # Transform new_matrix to a Laplacian matrix
    new_laplacian = laplacian(pad_adjacency_matrices([amat])[0])

    # Flatten the Laplacian matrix
    new_laplacian_flattened = new_laplacian.flatten()

    # Reshape the flattened matrix to fit the input shape of the KMeans model
    new_laplacian_reshaped = new_laplacian_flattened.reshape(1, -1)

    # Use the KMeans model to predict the cluster of the new matrix
    cluster_label = loaded_model.predict(new_laplacian_reshaped)
    
    print(cluster_label[0])
