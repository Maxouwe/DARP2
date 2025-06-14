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

#counts shared words
def getNormalizedSharedWords(row):
    intersect = set.intersection(set(row['normalized_title']), set(row['normalized_st']))
    return len(intersect)

def getSharedWords(row):
    intersect = set.intersection(set(row['product_title']), set(row['search_term']))
    return len(intersect)

def getNormalizedSharedWordsPD(row):
    intersect = set.intersection(set(row['normalized_pd']), set(row['normalized_st']))
    return len(intersect)


#looks at the set of all the words that search_term and product_title dont share
def getWordDifferenceRatioNormalized(row):
    title = set(row['normalized_title'])
    st = set(row['normalized_st'])
    diff = set.union(set.difference(title, st), set.difference(st, title))
    un = set.union(title, st)
    return len(diff)/len(un)

def getWordDifferenceRatioNormalizedPD(row):
    desc = set(row['normalized_pd'])
    st = set(row['normalized_st'])
    diff = set.union(set.difference(desc, st), set.difference(st, desc))
    un = set.union(desc, st)
    return len(diff)/len(un)

def getWordDifferenceRatio(row):
    title = set(row['product_title'])
    st = set(row['search_term'])
    diff = set.union(set.difference(title, st), set.difference(st, title))
    un = set.union(title, st)
    return len(diff)/len(un)


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

#needed to weight the proximity score
def getNrOfSharedWords(row):
    shared_words = 0
    for w in row['normalized_st']:
        if w in row['position_lists']:
            shared_words+= 1
    return shared_words

def getProximityScoreRow(row):
    k = getNrOfSharedWords(row)
    search_terms = list(row['normalized_st'])
    score_weight = k/len(search_terms)
    pos_lists = dict(row['position_lists'])
    heads = list()
    shortestInterval = 2**61
    finished = False
    while not finished:
        for term in search_terms:
            heads.append([pos_lists[term].pop(0), term])
            heads.sort()
        if heads.last - heads.first < shortestInterval:
            shortestInterval = heads.last - heads.first
        if len(pos_lists[term]) < 1:
            finished = True
    return shortestInterval





