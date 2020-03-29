import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def top_tfidf(docs, n):
    """
    get top n terms with highest TF-IDF value 
    from some docs
    
    to retrieve all, put n=-1
    
    inspired by https://gist.github.com/StevenMaude/ea46edc315b0f94d03b9
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(docs)

    transformed = vectorizer.transform(docs)

    scores = zip(vectorizer.get_feature_names(), np.asarray(transformed.sum(axis=0)).ravel())
    top_term_with_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return top_term_with_scores[:n]

def top_doc(df, doc_col, score_col, m):
    """
    get top n most relevant docs,
    to retrieve all, put m=-1
    doc_col = df column that contains text
    score_col = df column that contains score
    
    """
    sorted_score = df[score_col].sort_values(ascending=False)
    top_sorted_score = df[score_col].sort_values(ascending=False).head(m)
    doc_idx = top_sorted_score.index
    top_docs = df.loc[doc_idx, doc_col]
    
    return top_docs

def top_term_from_top_doc(df, doc_col, score_col, m, n):
    """
    get top m terms from n most relevant docs
    """
    docs = top_doc(df, doc_col, score_col, m)
    top_term = top_tfidf(docs,n)
    return top_term
