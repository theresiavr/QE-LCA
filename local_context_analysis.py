# -*- coding: utf-8 -*-
"""
Functions to get top n terms from top m documents
or simply just to fetch the top n terms or the top m docs.

Top n terms are obtained using the highest TF-IDF values.
Top m documents are obtained using some kind of relevance score.

"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def top_tfidf(docs, n):
    """
    Get top n terms with highest TF-IDF value 
    from some docs.
       
    Args:
        docs (list/Series): Text document in the form of a nested list or pandas.Series
        n (int): Number of terms. To get all possible terms, put n=-1

    Returns:
        top_term_with_scores: returns top n terms with the highest TF-IDF values, 
        together with the TF-IDF values

    inspired by https://gist.github.com/StevenMaude/ea46edc315b0f94d03b9
    """
    #initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    #Fit vectorizer to the documents
    vectorizer.fit(docs)

    #Get TF-IDF values from the documents
    transformed = vectorizer.transform(docs)

    #Get the TF-IDF value for each word/vocabulary
    scores = zip(vectorizer.get_feature_names(), np.asarray(transformed.sum(axis=0)).ravel())
    
    #Sort terms based on highest TF-IDF value
    top_term_with_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    #Return top n terms with the highest TF-IDF value
    return top_term_with_scores[:n]

def top_doc(df, doc_col, score_col, m):
    """
    Get top m most relevant docs 
    based on some relevance scores.
    
    Args:
        df = pandas.DataFrame object that contains the documents and their respective relevance scores
        doc_col (str) = Name of DataFrame column that contains text
        score_col (str) = Name of DataFrame column that contains score
        m (int) = Number of top docs to get. To retrieve all, put m=-1
    
    Returns:
        top_docs: returns top m documents with the highest relevance scores
    """
    #Sort score
    sorted_score = df[score_col].sort_values(ascending=False)

    #Get top m scores
    top_sorted_score = sorted_score.head(m)

    #Get index of top m scores
    doc_idx = top_sorted_score.index

    #Get top m documents based on the scores
    top_docs = df.loc[doc_idx, doc_col]
    
    #Return top m documents based on relevance scores
    return top_docs

def top_term_from_top_doc(df, doc_col, score_col, m, n):
    """
    Get top n terms from m most relevant docs

    
    Args:
        df = pandas.DataFrame object that contains the documents and their respective relevance scores
        doc_col (str) = Name of DataFrame column that contains text
        score_col (str) = Name of DataFrame column that contains score
        m (int) = Number of top docs to get. To retrieve all, put m=-1
    
    Returns:
        top_term: returns top n terms with the highest TF-IDF values, 
        together with the TF-IDF values
    """

    #Get top docs
    docs = top_doc(df, doc_col, score_col, m)

    #Get top terms
    top_term = top_tfidf(docs,n)

    #Return top terms
    return top_term
