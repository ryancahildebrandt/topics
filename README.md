# Explainable Topic Extraction with Decision Trees

---

[*Open*](https://gitpod.io/#https://github.com/ryancahildebrandt/topics) *in gitpod*

## Purpose
This project is a collection of a couple different experiments and utilities for topic extraction, including:
- Comparing popular topic extraction libraries on different kinds of documents
- A simple implementation of decision trees to explain group membership for clustered texts
- Some additional utilites for explaining decision tree rules based on bag of word embedding features
- Plotting utilities for derived clusters and/or their assigned topics

---

## Datasets
- [20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups)
- [Bitext Customer Support](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants)
- [ShopperSentiments](https://www.kaggle.com/datasets/nelgiriyewithana/shoppersentiments)
- [Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset)

### Topic Extractors
- [yake](https://github.com/LIAAD/yake)
- [pytextrank](https://spacy.io/universe/project/spacy-pytextrank)
- [lda](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
- [nmf](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [word count](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [tfidf score](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)
- [decision tree](./explain_with_dt.py)

---

## Outputs
- The results [report](./report.md) outlining approach and findings
- The [outputs](./outputs) folder, which contains breakdowns of dataset topics, topic evaluations, and tests of group differences
- [Decision tree](./explain_with_dt.py) topic extraction implementation