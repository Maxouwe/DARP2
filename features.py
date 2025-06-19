import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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

def correctPrediction(row):
    val = row['predicted_relevance'] == row['relevance']
    if val:
        return 1
    else:
        return 0

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
    freqs = dict(row['term_freqs'])
    scores = []
    for term in intersect:
        scores.append(idfdf[idfdf['term']==term]['idfscore'][0] * freqs[term])
    return scores.sum()

#needed to weight the proximity score
def getNrOfSharedWords(row):
    shared_words = 0
    for w in set(row['normalized_st']):
        if w in row['position_lists']:
            shared_words+= 1
    return shared_words


def getProximityScoreRow(row):
    
    search_terms = list(set(row['normalized_st']))
    pos_lists = dict(row['position_lists'])

    currentInterval = list()
    shortestLen = len(row['normalized_pd'])
    k = getNrOfSharedWords(row)
    if k == 0:
        return 0
    if k == 1:
        return 1/np.sqrt(len(row['normalized_pd'])/2)
    for term in search_terms:
        if term in pos_lists:
            currentInterval.append([pos_lists[term].pop(0), term])
            currentInterval.sort()
    while True:
        if len(pos_lists[currentInterval[0][1]]) < 1:
            if (currentInterval[-1][0] - currentInterval[0][0] < shortestLen):
                shortestLen = currentInterval[-1][0] - currentInterval[0][0]
            break
        p = [pos_lists[currentInterval[0][1]].pop(0), currentInterval[0][1]]
        q = [currentInterval[1][0], currentInterval[1][1]]
        
        if p[0] > currentInterval[-1][0]:
            if currentInterval[-1][0] - currentInterval[0][0] < shortestLen:
                shortestLen = currentInterval[-1][0] - currentInterval[0][0]
                currentInterval.pop(0)
                currentInterval.append(p)
        else:
            currentInterval.pop(0)
            if p[0] < q[0]:
                currentInterval.insert(0, p)
            else:
                currentInterval.insert(0, q)
            currentInterval.sort()
    return 1/np.sqrt(shortestLen)

def getKeywordDensity(row):
    k = getNrOfSharedWords(row)
    if k == 0:
        return 0
    count = 0
    for w in row['normalized_pd']:
        if w in row['normalized_st']:
            count += 1
    return count/k

def get_mean_vector(model, tokens):
    vecs = []
    for token in tokens:
        if token in model:
            vecs.append(model[token])
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)


def add_vector_similarities(df, model):
    title_sims = []
    desc_sims = []

    for _, row in df.iterrows():
        query_vec = get_mean_vector(model, row['normalized_st']).reshape(1, -1)
        title_vec = get_mean_vector(model, row['normalized_title']).reshape(1, -1)
        desc_vec = get_mean_vector(model, row['normalized_pd']).reshape(1, -1)

        title_sim = cosine_similarity(query_vec, title_vec)[0][0]
        desc_sim = cosine_similarity(query_vec, desc_vec)[0][0]

        title_sims.append(title_sim)
        desc_sims.append(desc_sim)

    df['TitleVecSim'] = title_sims
    df['DescriptionVecSim'] = desc_sims

    return df

#calculates how early the query terms in the description
#the earlier the better
#averages over all 
def averageEarlyScore(row):
    k = getNrOfSharedWords(row)
    scores = []
    for term in set(row['normalized_st']):
        for i in range(0, len(row['normalized_pd'])):
            if row['normalized_pd'][i] == term:
                scores.append(i)
                break
    if k > 0:
        return sum(scores)/k/len(row['normalized_pd'])
    else:
        return len(row['normalized_pd'])
    
def minimumEarlyScore(row):
    score = len(row['normalized_pd'])
    for term in set(row['normalized_st']):
        for i in range(0, len(row['normalized_pd'])):
            if i > score:
                break
            if row['normalized_pd'][i] == term:
                score = i
                break
    return score
    