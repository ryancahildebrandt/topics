#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 08:36:44 PM EDT 2024
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.api import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def write_tukeys(eval_df, name):
	for metric in ["exact_appearance", "shared_vocabulary", "semantic_similarity"]:
			temp_df = eval_df.query(f"eval_metric == '{metric}'")
			temp_df[["topic_source", "score"]].groupby("topic_source").mean(numeric_only = True).sort_values("score").to_csv(f"outputs/{name}_{metric}_means.csv")
			out = str(pairwise_tukeyhsd(endog = temp_df["score"].replace(np.nan, 0.), groups = temp_df["topic_source"]).summary())
			with open(f"outputs/{name}_{metric}_tukey.txt", "w") as f:
				f.write(out)

def write_aovs(eval_df, name):
	model = ols("score ~ C(topic_source)", data = eval_df).fit()
	with open(f"outputs/{name}_ols.txt", "w") as f:
		f.write(str(model.summary()))
	with open(f"outputs/{name}_aov.txt", "w") as f:
		f.write(str(anova_lm(model)))

def process_eval_df(in_df):
	out = pd.melt(
		in_df,
		id_vars = ["cluster", "dataset", "embeddings"],
		var_name = "topic_source",
		).rename(columns = {"value" : "score"})
	out["eval_metric"] = out["topic_source"].str.extract(r"([a-z]*_[a-z]*)$")
	out["topic_source"] = out["topic_source"].str.replace("topics_", "").str.extract(r"([a-z_]*)_[a-z]*_[a-z]*$")
	return out

def compare_dataset_topics(dataset):
	name = dataset["name"]
	dfs = []
	for k, v in dataset["evals"].items():
		df = pd.DataFrame(v).iloc[:, 1:]
		df["dataset"] = name
		df["embeddings"] = k
		dfs.append(df)	
	
	df = pd.concat(dfs).reset_index(names = "cluster")
	df = process_eval_df(df)
	
	write_tukeys(df, name)
	write_aovs(df, name)

	return df

