import json
import networkx as nx
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import random
import logging
import os
import javalang
import traceback

class JavaTDG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.classnames = set()

    def add_node(self, node_id, node_type, name, line_number=None, nullable=False, actual_type=None):
        self.graph.add_node(node_id, attr={'type': node_type, 'name': name, 'line_number': line_number, 'nullable': nullable, 'actual_type': actual_type})

    def add_edge(self, from_node, to_node, edge_type):
        self.graph.add_edge(from_node, to_node, type=edge_type)
        self.graph.add_edge(to_node, from_node, type=f"reverse_{edge_type}")

    def add_classname(self, classname):
        self.classnames.add(classname)

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))

class NodeIDMapper:
    def __init__(self):
        self.id_to_int = {}
        self.int_to_id = {}
        self.counter = 0

    def get_int(self, node_id):
        if node_id not in self.id_to_int:
            self.id_to_int[node_id] = self.counter
            self.int_to_id[self.counter] = node_id
            self.counter += 1
        return self.id_to_int[node_id]

    def get_id(self, node_int):
        return self.int_to_id.get(node_int, None)

node_id_mapper = NodeIDMapper()

# Define mappings globally
type_mapping = {'class': 0, 'method': 1, 'field': 2, 'parameter': 3, 'variable': 4, 'literal': 5}

def extract_features(attr):
    if attr is None:
        logging.warning("Encountered NoneType for attr. Using default values.")
        return [0.0, 0.0]  # Default feature vector with type_id and nullable

    node_type = attr.get('type', '')
    nullable = float(attr.get('nullable', 0))

    type_id = type_mapping.get(node_type, len(type_mapping))

    return [float(type_id), nullable]

def load_tdg_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        tdg = JavaTDG()
        tdg.graph = nx.node_link_graph(data)
        return preprocess_tdg(tdg)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {json_path}: {e}")
        return ([], [], [], [], [])  # Return empty placeholders if there's an error
    except Exception as e:
        logging.error(f"Error processing {json_path}: {e}")
        return ([], [], [], [], [])  # Handle other errors similarly

def balance_dataset(features, labels, node_ids, adjacency_matrix):
    pos_indices = [i for i, label in enumerate(labels) if label == 1]
    neg_indices = [i for i, label in enumerate(labels) if label == 0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        logging.warning("Cannot balance dataset with no positive or negative examples.")
        return features, labels, node_ids, adjacency_matrix

    random.shuffle(neg_indices)
    selected_neg_indices = neg_indices[:len(pos_indices)]

    selected_indices = pos_indices + selected_neg_indices
    random.shuffle(selected_indices)

    selected_features = features[selected_indices]
    selected_labels = labels[selected_indices]
    selected_node_ids = node_ids[selected_indices]
    selected_adjacency_matrix = adjacency_matrix[selected_indices, :][:, selected_indices]

    return selected_features, selected_labels, selected_node_ids, selected_adjacency_matrix

def preprocess_tdg(tdg):
    features = []
    labels = []
    node_ids = []
    prediction_node_ids = []
    all_node_ids = list(tdg.graph.nodes)

    if len(all_node_ids) == 0:
        return np.zeros((1, 2)), np.zeros((1,)), np.zeros((1,)), np.zeros((1, 1)), []

    node_id_map = {}  # Map node IDs to indices
    for idx, node_id in enumerate(all_node_ids):
        node_id_map[node_id] = idx

    num_nodes = len(all_node_ids)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for node_id in all_node_ids:
        attr = tdg.graph.nodes[node_id].get('attr', {})
        feature_vector = extract_features(attr)
        features.append(feature_vector)
        node_index = node_id_map[node_id]
        node_ids.append(node_index)

        # Map node IDs to indices for consistent mapping
        node_id_mapper.get_int(node_id)  # Ensure the node ID is mapped

        if attr.get('type') in ['method', 'field', 'parameter']:
            labels.append(float(attr.get('nullable', 0)))
            prediction_node_ids.append(node_index)

    for from_node, to_node in tdg.graph.edges():
        from_idx = node_id_map.get(from_node)
        to_idx = node_id_map.get(to_node)
        if from_idx is not None and to_idx is not None:
            adjacency_matrix[from_idx, to_idx] = 1.0

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    node_ids = np.array(node_ids, dtype=np.int32)

    if features.size == 0 or adjacency_matrix.size == 0:
        logging.warning("Skipping empty or invalid graph.")
        return np.zeros((1, 2)), np.zeros((1,)), np.zeros((1,)), np.zeros((1, 1)), []

    return features, labels, node_ids, adjacency_matrix, prediction_node_ids

def data_generator(file_list, balance=False, is_tdg=True):
    if is_tdg:
        # Training: Process pre-extracted graphs
        for file_path in file_list:
            try:
                result = load_tdg_data(file_path)
                if len(result) != 5:
                    logging.error(f"Graph from {file_path} returned {len(result)} values. Expected 5. Skipping this graph.")
                    continue

                features, labels, node_ids, adjacency_matrix, prediction_node_ids = result

                features = np.array(features, dtype=np.float32)
                adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)

                if features.size == 0 or adjacency_matrix.size == 0:
                    logging.warning(f"Skipping empty or invalid graph in file: {file_path}")
                    continue

                if balance:
                    features, labels, prediction_node_ids, adjacency_matrix = balance_dataset(features, labels, prediction_node_ids, adjacency_matrix)

                yield (features, adjacency_matrix, prediction_node_ids), labels
            except Exception as e:
                logging.error(f"Error processing graph in file {file_path}: {e}")
                continue
    else:
        # Prediction logic (if needed)
        pass

def create_tf_dataset(file_list, batch_size, balance=False, is_tdg=True):
    def generator():
        for (features, adjacency_matrix, prediction_node_ids), labels in data_generator(file_list, balance, is_tdg):
            features = np.array(features, dtype=np.float32)
            adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)

            prediction_mask = np.zeros(features.shape[0], dtype=bool)
            prediction_mask[prediction_node_ids] = True

            # Extract labels corresponding to the prediction_mask
            all_labels = np.zeros((features.shape[0], 1), dtype=np.float32)
            all_labels[prediction_node_ids] = labels.reshape(-1, 1)
            masked_labels = all_labels[prediction_mask]

            yield (features, adjacency_matrix, prediction_mask), masked_labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (tf.TensorSpec(shape=(None, 2), dtype=tf.float32),  # Node features
             tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # Adjacency matrix
             tf.TensorSpec(shape=(None,), dtype=tf.bool)),  # Prediction mask
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # Labels
        )
    )

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            (tf.TensorShape([None, 2]),  # Node features
             tf.TensorShape([None, None]),  # Adjacency matrix
             tf.TensorShape([None])),  # Prediction mask
            tf.TensorShape([None, 1])  # Labels
        ),
        padding_values=(
            (tf.constant(0.0),  # Padding value for features
             tf.constant(0.0),  # Padding value for adjacency matrix
             tf.constant(False)),  # Padding value for prediction mask
            tf.constant(0.0)  # Padding value for labels
        )
    )
    return dataset

# ... (Other functions like process_java_file remain unchanged or are updated similarly)
