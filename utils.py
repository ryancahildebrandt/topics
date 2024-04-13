#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 01:44:48 PM EDT 2024
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import pickle
import pandas as pd
import umap
import umap.plot
import bokeh.plotting
# dataset prep
def create_dataset(texts, name, limit = None):
	return {
		"name" : name,
		"texts" : texts[:limit],
		"topics" : {},
		"evals" : {},
		}

def write_dataset(dataset):
	name = dataset["name"]
	with open(f"data/{name}.pickle", "wb") as f:
		pickle.dump(dataset, f)
	print(f"saved {name} dataset")

def write_dataset_topics(dataset):
	name = dataset["name"]
	for k, v in dataset["topics"].items():
		pd.DataFrame(v).to_csv(f"outputs/{name}_{k}_topics.tsv", sep = "\t")
		print(f"saved {name}_{k}_topics.tsv")

def write_dataset_evals(dataset):
	name = dataset["name"]
	for k, v in dataset["evals"].items():
		df = pd.DataFrame(v)

		df.to_csv(f"outputs/{name}_{k}_evals.tsv", sep = "\t")
		print(f"saved {name}_{k}_evals.tsv")

		df.describe().to_csv(f"outputs/{name}_{k}_evals_agg.tsv", sep = "\t")
		print(f"saved {name}_{k}_evals_agg.tsv")

# topic plotting
def plot_topics(x, y , labels, outfile):
	umap.plot.output_file(outfile)
	mapper = umap.UMAP(n_neighbors = 5).fit(x, y = y)
	hover = pd.DataFrame(
		{
			"label": y,
			"topics" : labels
		}
	)
	p = umap.plot.interactive(
		mapper,
		labels = labels,
		hover_data = hover,
		point_size = 10
		)
	bokeh.plotting.save(p)
