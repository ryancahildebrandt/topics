#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 08:02:50 PM EDT 2024
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import pandas as pd
import pickle

from compare_topics import compare_dataset_topics, process_eval_df, write_aovs, write_tukeys
from embed_and_cluster import add_embeddings_and_clusters_to_dataset, use_embedder
from evaluate_topics import evaluate_dataset_topics
from extract_topics import add_topics_to_dataset
from readin import bitext, news_short, news_long, review, sentiment
from utils import plot_topics, create_dataset

# load datasets
from_pickles = True
if from_pickles:
	with open("data/bitext.pickle", "rb") as f:
		bitext = pickle.load(f)
	with open("data/news_short.pickle", "rb") as f:
		news_short = pickle.load(f)
	with open("data/news_long.pickle", "rb") as f:
		news_long = pickle.load(f)
	with open("data/review.pickle", "rb") as f:
		review = pickle.load(f)
	with open("data/sentiment.pickle", "rb") as f:
		sentiment = pickle.load(f)
else:
	bitext = create_dataset(bitext, "bitext", 10000)
	news_short = create_dataset(news_short, "news_short", 10000)
	news_long = create_dataset(news_long, "news_long", 10000)
	review = create_dataset(review, "review", 10000)
	sentiment = create_dataset(sentiment, "sentiment", 10000)

overall = []
for dataset in [bitext, news_short, news_long, review, sentiment]:
	#add_embeddings_and_clusters_to_dataset(dataset)
	add_topics_to_dataset(dataset)
	evaluate_dataset_topics(dataset)
	overall.append(compare_dataset_topics(dataset))

df = pd.DataFrame({"topics" : bitext["topics"]["use"]["topics_pytextrank"], "cluster" : bitext["topics"]["use"]["id"]}).explode("topics")
plot_topics(
	x = bitext["use"],
	y = bitext["use_cluster"],
	labels = bitext["texts"],
	outfile = f"outputs/bitext_use_texts_fig.html"
)
plot_topics(
	x = use_embedder(df["topics"]),
	y = df["cluster"].tolist(),
	labels = df["topics"].tolist(),
	outfile = f"outputs/bitext_use_pytextrank_topics_fig.html"
)

overall = pd.concat(overall)
write_tukeys(overall, "overall")
write_aovs(overall, "overall")
