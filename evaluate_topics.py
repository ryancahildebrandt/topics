#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 08:41:22 PM EDT 2024 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm as tq
import tensorflow_hub as hub

from utils import write_dataset, write_dataset_evals

use_embedder = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2")

# topic evaluation
def eval_exact_appearance(texts_list, topics_list):
	out = []

	for texts, topics in zip(texts_list, topics_list):
		text = " ".join(texts)
		topic_in_text = [topic in text for topic in topics]
		out.append(np.mean(topic_in_text))

	return out

def eval_shared_vocabulary(texts_list, topics_list):
	out = []

	for texts, topics in zip(texts_list, topics_list):
		text_vocab = set(" ".join(texts).split())
		topic_vocab = set(" ".join(topics).split())
		shared_vocab = [vocab in text_vocab for vocab in topic_vocab]
		out.append(np.mean(shared_vocab))

	return out

def eval_semantic_similarity(embs_list, topics_list):
	out = []

	for texts_embs, topics in zip(embs_list, topics_list):
		topics_embs = use_embedder(topics) if topics else use_embedder([""])
		distances = pairwise_distances(texts_embs, topics_embs, metric = "cosine")
		similarity = 1 - np.mean(distances)
		out.append(similarity)

	return out

def eval_topic_set(topic_set, embs_arr, all_texts):
	grouped_texts = topic_set["texts"]
	out = {"texts" : grouped_texts}
	embs = []

	for group in grouped_texts:
		inds = [all_texts.index(text) for text in group]
		embs.append(embs_arr[inds])
		
	for k, v in tq(topic_set.items(), desc = "evaluating topics"):
		if "topics" in k:
			topics = v
			out[f"{k}_exact_appearance"] = eval_exact_appearance(grouped_texts, topics)
			out[f"{k}_shared_vocabulary"] = eval_shared_vocabulary(grouped_texts, topics)
			out[f"{k}_semantic_similarity"] = eval_semantic_similarity(embs, topics)

	return out

def evaluate_dataset_topics(dataset):
	name = dataset["name"]
	dataset["evals"] = {
		"count" : eval_topic_set(dataset["topics"]["count"], dataset["count"], dataset["texts"]),
		"tfidf" : eval_topic_set(dataset["topics"]["tfidf"], dataset["tfidf"], dataset["texts"]),
		"use" : eval_topic_set(dataset["topics"]["use"], dataset["use"], dataset["texts"])
		}	
	write_dataset(dataset)
	write_dataset_evals(dataset)
	print(f"evaluated topics for {name} dataset")
