#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 07:20:00 PM EST 2024
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import re
import pandas as pd
import itertools
from sklearn.datasets import fetch_20newsgroups

# texts
bitext = pd.read_csv("./data/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv")["utterance"].tolist()#[:100]
sentiment = pd.read_csv("./data/sentimentdataset.csv")["Text"].tolist()
review = pd.read_csv("./data/TeePublic_review.csv", encoding = "latin-1").astype(str)["review"].tolist()
news_long = fetch_20newsgroups().data
news_short = map(lambda x: re.split("\.\s+", x.replace("\n", " ").strip()), news_long)
news_short = list(itertools.chain.from_iterable(news_short))
