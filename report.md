# Explainable Topic Extraction with Decision Trees
### *Using decision tree classification to explain text clusters, with benchmarking against popular topic extractors*

---

## Purpose
This project is a collection of a couple different experiments and utilities for topic extraction, including:
- Comparing popular topic extraction libraries on different kinds of documents
- A simple implementation of decision trees to explain group membership for clustered texts
- Some additional utilites for explaining decision tree rules based on bag of word embedding features
- Plotting utilities for derived clusters and/or their assigned topics

---

## Approach
The processing pipeline was as follows for each available dataset, embedding, and topic extractor:
- Select text documents from dataset
- Calculate embeddings
- Cluster using HDBSCAN algorithm
- Extract cluster topics
- Calculate evaluation metrics for each topic

### Datasets
- [20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups), for longer documents and split into short and long document variants
- [Bitext Customer Support](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants)
- [ShopperSentiments](https://www.kaggle.com/datasets/nelgiriyewithana/shoppersentiments)
- [Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset)

### Embeddings
- [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)
- [Count](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [Universal Sentence Embedder](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)

### Topic Extractors
- [yake](https://github.com/LIAAD/yake)
- [pytextrank](https://spacy.io/universe/project/spacy-pytextrank)
- [lda](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
- [nmf](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [word count](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [tfidf score](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)
- [decision tree](./explain_with_dt.py)

---

## Implementation 
Of the models/techniques listed above, 2 were implemented specifically for this project and are explained below

### Evaluation Metrics
These metrics were used for a quick and reproducible comparison of topic quality across clusters and topic extractors, but don't constitute a comprehensive evaluation tool without qualitative comparisons alongside. These metrics, for example, do not capture how comprehensively the topics apply to the texts, nor how many of the texts contain the returned topics.
- Shared Vocabulary
	- % of terms returned in topics also present in documents
- Exact Appearances
	- % of returned topics appearing verbatim in documents
- Semantic Similarity
	- Average pairwise cosine similarity between returnted topics and documents

### Decision Tree Topic Extractor
The process of topic extraction via decision tree broadly consists of the following steps:
- Identify the cluster to undergo topic extraction (target cluster)
- Embed the target cluster documents using bag of words embeddings, noting the corresponding term for each embedding feature
- Train a decision tree to label the entire text document dataset as either belonging or not belonging to the target cluster
- Calculate feature importances for the trained decision tree
- Select those features with the highest importance and/or those appearing in the target cluster vocabulary as the extracted topics

---

## Results

### Eval Metrics & Topic Extractors
For exact appearance and shared vocabulary, all topic extractors tended to perform pretty well, with mean scores between .75 and 1. It's worth noting that this is a bit of a moot point when looking at the decision tree topics, as they cannot return topics that don't appear in the original documents. Beyond the decision tree topics, the next best performers were pytextrank and yake. The evaluation means tend to me much lower when looking at semantic similarity, but this is calculated differently from the other two metrics so isn't necessarily comparable. In fact, looking at the overall pattern of how well each topic extractor performed, we see a very similar pattern of the highest performers being decision tree based, folowed by pytextrank and yake.

### Extracted Topics
Qualitatively, the topic extractors tend to differ in the following ways
- **Yake** returns short phrases which tend to be fairly interpretable on their own, as well as topics that are shared across a large number of documents
- **Pytextrank** topics tend to be around 1-3 words and less interpretable than yake. It tends toward topics that are more varied, even if not shared by majority of documents
- **Count and tfidf** approaches return single words present in most documents
- **LDA and NMF** also returns single words present in most documents, and possibly shows more stopword topics
- **Decision tree** topics are less complete than simple bag of words topics, but more relevant to cluster membership
