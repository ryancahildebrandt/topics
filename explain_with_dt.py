#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 04:40:12 PM EDT 2024
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import sklearn
from sklearn import tree
import numpy as np

# decision tree
# these utilities were based in part on the examples from sklearn documentation: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
def train_discriminator_tree(embedding_array: np.ndarray, cluster_labels: np.ndarray, target_cluster: int, features: list, accuracy: bool = False) -> sklearn.tree.DecisionTreeClassifier:
	"""
	Trains a simple decision tree to label a set of datapoints as belonging/not belonging to one group in a clustered dataset

	Args:
		embedding_array (np.ndarray): embedding vectors for each observation in the dataset
		cluster_labels (np.ndarray): cluster label for each observation in the dataset
		target_cluster (int): target cluster label, observations will be labeled as a member or non member of this cluster
		features (list): terms corresponding to each element of the embedding vector
		accuracy (bool, optional): whether to return training accuracy score with trained discriminator tree. Defaults to False.

	Returns:
		sklearn.tree.DecisionTreeClassifier: trained decision tree
	"""
	dtc = tree.DecisionTreeClassifier()
	x = embedding_array
	y = [1 if i == target_cluster else 0 for i in cluster_labels]
	dtc.fit(x, y)
	out = dtc

	if accuracy:
		out = [dtc, dtc.score(x, y)]
	
	return out

def show_tree_structure(discriminator_tree: sklearn.tree.DecisionTreeClassifier, features: list) -> str:
	"""
	Prints the structure of the provided decision tree

	Args:
		discriminator_tree (sklearn.tree.DecisionTreeClassifier): trained decision tree
		features (list): terms corresponding to each element of the embedding vector

	Returns:
		str: tree structure diagram
	"""
	return tree.export_text(discriminator_tree, feature_names = features)

def important_features_for_tree(discriminator_tree: sklearn.tree.DecisionTreeClassifier, features: list, importances: bool = False) -> list:
	"""
	Returns features who's inclusion in a document is an indicator of group membership

	Args:
		discriminator_tree (sklearn.tree.DecisionTreeClassifier): trained decision tree
		features (list): terms corresponding to each element of the embedding vector
		importances (bool, optional): whether to return feature importances with terms. Defaults to False.

	Returns:
		list: extracted topics
	"""
	imps = discriminator_tree.tree_.compute_feature_importances()
	kw = [[feature, importance] for feature, importance in zip(features, imps) if importance != 0.0]
	out = sorted(kw, key = lambda x: x[1], reverse = True)

	if not importances:
		out = [i[0] for i in out]

	return out

def decisions_for_cluster(discriminator_tree: sklearn.tree.DecisionTreeClassifier, texts: list, embedding_array: np.ndarray, cluster_labels: np.ndarray, target_cluster: int, features: list) -> None:
	"""
	Prints the specific decisions used to classify the provided datapoints with the given decision tree

	Args:
		discriminator_tree (sklearn.tree.DecisionTreeClassifier): trained decision tree
		texts (list): documents to pass to the decision tree
		embedding_array (np.ndarray): embedding vectors for each observation in the dataset
		cluster_labels (np.ndarray): cluster label for each observation in the dataset
		target_cluster (int): target cluster label, observations will be labeled as a member or non member of this cluster
		features (list): terms corresponding to each element of the embedding vector
	"""
	target_ids = [index for index, text, cluster in zip(range(len(texts)), texts, cluster_labels) if cluster == target_cluster]
	decision_paths = discriminator_tree.decision_path(embedding_array)
	leaf_id = discriminator_tree.apply(embedding_array)
	feature = discriminator_tree.tree_.feature
	threshold = discriminator_tree.tree_.threshold

	for target_id in target_ids:
		text = texts[target_id]
		node_index = decision_paths.indices[decision_paths.indptr[target_id] : decision_paths.indptr[target_id + 1]]
		print(f"\n\nRules used to predict sample {target_id}, '{text}':")
		for node_id in node_index:
			# continue to the next node if it is a leaf node
			if leaf_id[target_id] == node_id:
				continue
			# check if value of the split feature for sample is below threshold
			if embedding_array[target_id, feature[node_id]] <= threshold[node_id]:
				presence = "is not"
			else:
				presence = "is"
			print(f"decision node {node_id} : {features[feature[node_id]]} = {embedding_array[target_id, feature[node_id]]}, {presence} present in sample")

def explain_decision_tree(discriminator_tree: sklearn.tree.DecisionTreeClassifier, features: list) -> None:
	"""
	Prints an explanation of the decisions made as a part of the decision tree, including relevant features

	Args:
		discriminator_tree (sklearn.tree.DecisionTreeClassifier): trained decision tree
		features (list): terms corresponding to each element of the embedding vector
	"""
	n_nodes = discriminator_tree.tree_.node_count
	children_left = discriminator_tree.tree_.children_left
	children_right = discriminator_tree.tree_.children_right
	threshold = discriminator_tree.tree_.threshold
	values = discriminator_tree.tree_.value

	node_depth = np.zeros(shape = n_nodes, dtype = np.int64)
	is_leaves = np.zeros(shape = n_nodes, dtype = bool)
	stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
	while len(stack) > 0:
		# `pop` ensures each node is only visited once
		node_id, depth = stack.pop()
		node_depth[node_id] = depth

		# If the left and right child of a node is not the same we have a split node
		is_split_node = children_left[node_id] != children_right[node_id]

		# If a split node, append left and right children and depth to `stack` so we can loop through them
		if is_split_node:
			stack.append((children_left[node_id], depth + 1))
			stack.append((children_right[node_id], depth + 1))
		else:
			is_leaves[node_id] = True

	print(f"Decision tree has {n_nodes} nodes and structure:")
	for i in range(n_nodes):
		if is_leaves[i]:
			print(f"{node_depth[i]*'  '}node {i} is a leaf node with value {values[i]}")
		else:
			print(f"{node_depth[i]*'  '}node {i} is a split node with value {values[i]}: go to node {children_left[i]} if {features[i]} <= {threshold[i]} else to node {children_right[i]}.")
