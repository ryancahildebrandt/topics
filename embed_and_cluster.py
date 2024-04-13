#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 08:05:03 PM EDT 2024 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import tensorflow_hub as hub
import more_itertools
import itertools
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import HDBSCAN
from tqdm import tqdm as tq

from utils import write_dataset

# embed
use_embedder = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2")

def batched_embeddings(texts):
	out = []
	texts = more_itertools.chunked(texts, 1000)
	for chunk in tq(texts, desc = "embedding with USE"):
		out.append(use_embedder(chunk))
	out = np.array(list(itertools.chain.from_iterable(out)))
	return out

def add_embeddings(dataset):
	tfidf = TfidfVectorizer(max_features = 512)
	count = CountVectorizer(max_features = 512)
	dataset["use"] = batched_embeddings(dataset["texts"])
	dataset["tfidf"] = tfidf.fit_transform(dataset["texts"])
	dataset["tfidf_vocab"] = tfidf.get_feature_names_out()
	dataset["count"] = count.fit_transform(dataset["texts"])
	dataset["count_vocab"] = count.get_feature_names_out()

# cluster
def add_clusters(dataset):
	dataset["use_cluster"] = HDBSCAN().fit_predict(dataset["use"])
	dataset["tfidf_cluster"] = HDBSCAN().fit_predict(dataset["tfidf"].toarray())
	dataset["count_cluster"] = HDBSCAN().fit_predict(dataset["count"].toarray())

# pipe
def add_embeddings_and_clusters_to_dataset(dataset):
	add_embeddings(dataset)
	add_clusters(dataset)
	write_dataset(dataset)
	print(f"processed embeddings and clusters for {dataset['name']} dataset")
