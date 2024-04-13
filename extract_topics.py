#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 06:43:12 PM EDT 2024 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import yake
import spacy
import pytextrank
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm as tq
import numpy as np
import itertools

from explain_with_dt import important_features_for_tree, train_discriminator_tree
from utils import write_dataset, write_dataset_topics

# topic extractors
def yake_topics_from_texts(texts, importances = False):
	ykw = yake.KeywordExtractor()
	text = " ".join(texts)
	kw = ykw.extract_keywords(text)
	#high scores are bad so 1-v
	kw = [(i[0], 1 - i[1]) for i in kw]
	out = sorted(kw, key = lambda x: x[1], reverse = True)
	
	if not importances:
		out = [i[0] for i in out]
	
	return out

def pytextrank_topics_from_texts(nlp, texts, importances = False):
	text = " ".join(texts)[:nlp.max_length]
	doc = nlp(text)
	kw = [(phrase.text, phrase.rank) for phrase in doc._.phrases]
	out = sorted(kw, key = lambda x: x[1], reverse = True)

	if not importances:
		out = [i[0] for i in out]

	return out

def lda_topics_from_embeddings(embedding_array, features, importances = False):
	lda = LatentDirichletAllocation(1)
	lda.fit(embedding_array)
	topic = lda.components_[0]
	kw = [(features[t], topic[t]) for t in topic.argsort()[::-1]]
	out = sorted(kw, key = lambda x: x[1], reverse = True)
	
	if not importances:
		out = [i[0] for i in out]

	return out[:10]

def nmf_topics_from_embeddings(embedding_array, features, importances = False):
	nmf = NMF(1)
	nmf.fit(embedding_array)
	topic = nmf.components_[0]
	kw = [(features[t], topic[t]) for t in topic.argsort()[::-1]]
	out = sorted(kw, key = lambda x: x[1], reverse = True)
	
	if not importances:
		out = [i[0] for i in out]

	return out[:10]

def count_topics_from_texts(texts, importances = False):
	text = " ".join(texts)
	vectorizer = CountVectorizer()
	try:
		vectorizer.fit([text])
		vocab = vectorizer.get_feature_names_out()
		counts = np.asarray(vectorizer.transform([text]).todense())[0]
		out = dict(sorted(zip(vocab, counts), key = lambda x: x[1], reverse = True))
		
		if not importances:
			out = list(out.keys())
	
	except ValueError as e:
		print(e)
		out = []
	
	return out[:10]

def tfidf_topics_from_texts(texts, importances = False):
	text = " ".join(texts)
	vectorizer = TfidfVectorizer(use_idf = False)
	try:
		vectorizer.fit([text])
		vocab = vectorizer.get_feature_names_out()
		counts = np.asarray(vectorizer.transform([text]).todense())[0]
		out = dict(sorted(zip(vocab, counts), key = lambda x: x[1], reverse = True))
				
		if not importances:
			out = list(out.keys())

	except ValueError as e:
		print(e)
		out = []
	
	return out[:10]

def tfidf_topics_from_corpus(embedding_array, cluster_labels, target_cluster, features):
	embs = np.asarray(embedding_array.todense())[cluster_labels == target_cluster]
	out = []
	for emb in embs:
		for e in features[emb.argsort()[::-1]][:5]:
			if e not in out:
				out.append(e)
	
	return out[:10]

# add to dataset
def add_topics_for_use(dataset, nlp):
	dataset["topics"]["use"] = {
		"id" : [],
		"texts" : [],
		"topics_yake" : [],
		"topics_pytextrank" : [],
		"topics_count" : [],
		"topics_tfidf_texts" : [],
		"topics_count_dt" : [],
		"count_dt_acc" : [],
		"topics_tfidf_dt" : [],
		"tfidf_dt_acc" : []
		}

	for label in tq(np.unique(dataset["use_cluster"])):
		texts = list(itertools.compress(dataset["texts"], dataset["use_cluster"] == label))
		vocab = set(" ".join(texts).split())
		count_dt, count_acc = train_discriminator_tree(
			embedding_array = dataset["count"],
			cluster_labels = dataset["use_cluster"],
			target_cluster = label,
			features = dataset["count_vocab"],
			accuracy = True
			)
		tfidf_dt, tfidf_acc = train_discriminator_tree(
			embedding_array = dataset["tfidf"],
			cluster_labels = dataset["use_cluster"],
			target_cluster = label,
			features = dataset["tfidf_vocab"],
			accuracy = True
			)
		
		dataset["topics"]["use"]["id"].append(label)
		dataset["topics"]["use"]["texts"].append(texts)
		dataset["topics"]["use"]["topics_yake"].append(yake_topics_from_texts(texts))
		dataset["topics"]["use"]["topics_pytextrank"].append(pytextrank_topics_from_texts(nlp, texts))
		dataset["topics"]["use"]["topics_count"].append(count_topics_from_texts(texts))
		dataset["topics"]["use"]["topics_tfidf_texts"].append(tfidf_topics_from_texts(texts))
		dataset["topics"]["use"]["topics_count_dt"].append([i for i in important_features_for_tree(count_dt, dataset["count_vocab"]) if i in vocab])
		dataset["topics"]["use"]["count_dt_acc"].append(count_acc)
		dataset["topics"]["use"]["topics_tfidf_dt"].append([i for i in important_features_for_tree(count_dt, dataset["tfidf_vocab"]) if i in vocab])
		dataset["topics"]["use"]["tfidf_dt_acc"].append(tfidf_acc)

def add_topics_for_count(dataset, nlp):
	dataset["topics"]["count"] = {
		"id" : [],
		"texts" : [],
		"topics_yake" : [],
		"topics_pytextrank" : [],
		"topics_lda" : [],
		"topics_nmf" : [],
		"topics_count" : [],
		"topics_tfidf_texts" : [],
		"topics_tfidf_corpus" : [],
		"topics_dt" : [],
		"dt_acc" : []
	}
	
	for label in tq(np.unique(dataset["count_cluster"])):
		cluster_labels = dataset["count_cluster"]
		embs = dataset["count"][cluster_labels == label]
		full_vocab = dataset["count_vocab"]
		texts = list(itertools.compress(dataset["texts"], cluster_labels == label))
		vocab = set(" ".join(texts).split())
		dt, acc = train_discriminator_tree(
			embedding_array = dataset["count"],
			cluster_labels = cluster_labels,
			target_cluster = label,
			features = full_vocab,
			accuracy = True
			)

		dataset["topics"]["count"]["id"].append(label)
		dataset["topics"]["count"]["texts"].append(texts)
		dataset["topics"]["count"]["topics_yake"].append(yake_topics_from_texts(texts))
		dataset["topics"]["count"]["topics_pytextrank"].append(pytextrank_topics_from_texts(nlp, texts))
		dataset["topics"]["count"]["topics_count"].append(count_topics_from_texts(texts))
		dataset["topics"]["count"]["topics_tfidf_texts"].append(tfidf_topics_from_texts(texts))
		dataset["topics"]["count"]["topics_lda"].append(lda_topics_from_embeddings(embedding_array = embs, features = full_vocab))
		dataset["topics"]["count"]["topics_nmf"].append(nmf_topics_from_embeddings(embedding_array = embs, features = full_vocab))
		dataset["topics"]["count"]["topics_tfidf_corpus"].append(tfidf_topics_from_corpus(embedding_array = dataset["tfidf"], cluster_labels = cluster_labels, target_cluster = label, features = full_vocab))
		dataset["topics"]["count"]["topics_dt"].append([i for i in important_features_for_tree(dt, vocab) if i in vocab])
		dataset["topics"]["count"]["dt_acc"].append(acc)

def add_topics_for_tfidf(dataset, nlp):
	dataset["topics"]["tfidf"] = {
		"id" : [],
		"texts" : [],
		"topics_yake" : [],
		"topics_pytextrank" : [],
		"topics_lda" : [],
		"topics_nmf" : [],
		"topics_count" : [],
		"topics_tfidf_texts" : [],
		"topics_tfidf_corpus" : [],
		"topics_dt" : [],
		"dt_acc" : []
	}

	for label in tq(np.unique(dataset["tfidf_cluster"])):
		cluster_labels = dataset["tfidf_cluster"]
		embs = dataset["tfidf"][cluster_labels == label]
		full_vocab = dataset["tfidf_vocab"]
		texts = list(itertools.compress(dataset["texts"], cluster_labels == label))
		vocab = set(" ".join(texts).split())
		dt, acc = train_discriminator_tree(
			embedding_array = dataset["tfidf"],
			cluster_labels = cluster_labels,
			target_cluster = label,
			features = full_vocab,
			accuracy = True
			)
		
		dataset["topics"]["tfidf"]["id"].append(label)
		dataset["topics"]["tfidf"]["texts"].append(texts)
		dataset["topics"]["tfidf"]["topics_yake"].append(yake_topics_from_texts(texts))
		dataset["topics"]["tfidf"]["topics_pytextrank"].append(pytextrank_topics_from_texts(nlp, texts))
		dataset["topics"]["tfidf"]["topics_count"].append(count_topics_from_texts(texts))
		dataset["topics"]["tfidf"]["topics_tfidf_texts"].append(tfidf_topics_from_texts(texts))
		dataset["topics"]["tfidf"]["topics_lda"].append(lda_topics_from_embeddings(embedding_array = embs, features = full_vocab))
		dataset["topics"]["tfidf"]["topics_nmf"].append(nmf_topics_from_embeddings(embedding_array = embs, features = full_vocab))
		dataset["topics"]["tfidf"]["topics_tfidf_corpus"].append(tfidf_topics_from_corpus(embedding_array = dataset["tfidf"], cluster_labels = cluster_labels, target_cluster = label, features = full_vocab))
		dataset["topics"]["tfidf"]["topics_dt"].append([i for i in important_features_for_tree(dt, vocab) if i in vocab])
		dataset["topics"]["tfidf"]["dt_acc"].append(acc)

def add_topics_to_dataset(dataset):
	name = dataset["name"]
	nlp = spacy.load("en_core_web_sm")
	nlp.add_pipe("textrank")
	
	add_topics_for_use(dataset = dataset, nlp = nlp)
	add_topics_for_count(dataset = dataset, nlp = nlp)
	add_topics_for_tfidf(dataset = dataset, nlp = nlp)
	
	write_dataset(dataset)
	write_dataset_topics(dataset)
	print(f"processed topics for {name} dataset")
