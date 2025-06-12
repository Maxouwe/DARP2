import pandas as pd
import numpy as np
#these are features executed per row
#to use these features you use d.apply(feature, axis=1, args=[arg1, arg2])
#where d is the data you want to use the feature on
#axis=1 means you apply the feature to every row
#and the array of args is are the parameters, aside from the row itself, that get passed to the function
#see main.py line 29 and preprocessFunctions.py line 126 for examples
def check_words(row):
    words_column1 = set(row['product_title'].lower().split())
    words_column2 = set(row['search_term'].lower().split())
    return int(words_column2.issubset(words_column1))

#gets the relevance rating according to ordinal logistic regression
def getPredictedRelevance(row):
    probs = [row['p1'], row['p2'], row['p3']]
    max = 0
    classification = -1
    for i in [0, 1, 2]:
        if probs[i] > max:
            max = probs[i]
            classification = i + 1
    return classification

def getQFScore(row, qfdf):
    intersect = set.intersection(set(row['normalized_title']), set(row['normalized_st']))
    scores = qfdf[qfdf['term'].isin(intersect)]['qfscore']
    return scores.sum()

def getTitleIDFScore(row, idfdf):
    intersect = set.intersection(set(row['normalized_title']), set(row['normalized_st']))
    scores = idfdf[idfdf['term'].isin(intersect)]['idfscore']
    return scores.sum()

def getPDIDFScore(row, idfdf):
    intersect = set.intersection(set(row['normalized_pd']), set(row['normalized_st']))
    scores = idfdf[idfdf['term'].isin(intersect)]['idfscore']
    return scores.sum()
