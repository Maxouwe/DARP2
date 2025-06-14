import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import string
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from ast import literal_eval
import numpy as np

#not needed if youve executed these once already
#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger_eng')


stop_words = set(stopwords.words('english'))
units = set(['x', 'ft', 'sq', 'lb', 'psi', 'inc', 'cu', 'h', 'w'])


#removes punctuation from given dataframe and columnname
def removePunctuation(df, columnName):
    df[columnName] = df[columnName].str.translate(str.maketrans('', '', string.punctuation))

#removes stopwords from a row from a dataframe
def removeStopWordsFromRow(row, columnName):
    filtered_sentence = []
    for w in row[columnName]:
        if w.lower() not in stop_words:
            filtered_sentence.append(w.lower())
    return filtered_sentence

#removes all stopwords from given dataframe and columnname
def removeStopWordsFromColumn(df, columnName):
    df[columnName] = df.apply(removeStopWordsFromRow, axis=1, args=[columnName])

#removes all numerbers and units
def removeUnitsAndNumbersFromRow(row, columnName):
    filtered_sentence = []
    for w in row[columnName]:
        if w.lower() not in units and not w.isnumeric():
            filtered_sentence.append(w.lower())
    return filtered_sentence

#removes all stopwords from given dataframe and columnname
def removeUnitsAndNumbersFromColumn(df, columnName):
    df[columnName] = df.apply(removeUnitsAndNumbersFromRow, axis=1, args=[columnName])

#turns string into list of strings 
def tokenizeRow(row, columnName):
    return word_tokenize(row[columnName])

def tokenizeColumn(df, columnName):
    df[columnName] = df.apply(tokenizeRow, axis=1, args=[columnName])

#next 4 functions are for lemmatization see google
def lemmatizeColumn(df, columnName):
    lemmatizer = WordNetLemmatizer()
    df[columnName] = df.apply(lemmatizeRow, axis=1, args=[columnName, lemmatizer])

def get_wordnet_pos_list(s):
    tags = pos_tag(s)
    return [get_wordnet_pos_word(w) for w in tags]

def get_wordnet_pos_word(tag):
    p_tag = tag[1]
    if p_tag.startswith('J'):
        return wordnet.ADJ
    elif p_tag.startswith('V'):
        return wordnet.VERB
    elif p_tag.startswith('N'):
        return wordnet.NOUN
    elif p_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'
    
def lemmatizeRow(row, columnName, lemmatizer):
    s = row[columnName]
    processed_sentence = []    
    pos_tags = get_wordnet_pos_list(s)
    for i in range(0, len(s)):
        processed_sentence.append(lemmatizer.lemmatize(s[i], pos_tags[i]))
    return processed_sentence

#make a small version product_description to speed up testing
def makeSmallProductDescriptionsTable(df):
    filtered_df = df[df['product_uid'] <= 100030]
    filtered_df.to_csv("resources/small_pd.csv")

#when tokenizing a string it actually gets turned into a string looking like "[tokens]"
#so we need to turn it into a actual list
def turnStringFieldToList(df, columnName):
    df[columnName] = df[columnName].apply(literal_eval)

#creates qfscores of all query terms
def createQFScores(df):
    #contains all current queryFrequencies
    queryFrequencies = dict()
    queryFrequencies['RQFMax'] = 0
    for i in df:
        df.apply(updateQueryFrequenciesRow, axis=1, args=[queryFrequencies])

    #dataframe containing all current qfscores
    qfdf = pd.DataFrame(columns=['term', 'qfscore'])
    for term in queryFrequencies:
        qfdf.loc[len(qfdf)] = [term, (queryFrequencies[term]+1)/(queryFrequencies['RQFMax'] + 1)]
    return qfdf

#used in createQFScores
def updateQueryFrequenciesRow(row, queryFrequencies):
    for token in row['normalized_st']:
        if token in queryFrequencies:
            queryFrequencies[token] += 1
        else:
            queryFrequencies[token] = 1
        if queryFrequencies[token] > queryFrequencies['RQFMax']:
            queryFrequencies['RQFMax'] = queryFrequencies[token]
    
         
def createIDFScores(df, columnName):
    documentFrequencies = dict()
    for i in df:
        df.apply(updateTermFrequenciesRow, axis=1, args=[documentFrequencies, columnName])
    idfdf = pd.DataFrame(columns=['term', 'idfscore'])
    N = len(df)
    for term in documentFrequencies:
        idfdf.loc[len(idfdf)] = [term, np.log(N/documentFrequencies[term])]
    return idfdf

def updateTermFrequenciesRow(row, documentFrequencies, columnName):
    #we dont want to increment for duplicates
    rowe = set(row[columnName])
    for token in rowe:
        if token in documentFrequencies:
            documentFrequencies[token] += 1
        else:
            documentFrequencies[token] = 1

def createPosLists(pddf):
    posdf = pd.DataFrame(columns = ['product_uid', 'position_lists'])
    #position_lists contains for every term in the normalized description
    #a list of its position occurrence
    pddf.apply(createPosListsRow, axis=1, args=[posdf])
    return posdf

def createPosListsRow(row, posdf):
    positions = dict()
    description = row['normalized_pd']
    for i in range(0, len(description)):
        word = description[i]
        if not word in positions:
            positions[word] = list()
            positions[word].append(i)
        else:
            positions[word].append(i)
    posdf.loc[len(posdf)] = [row['product_uid'], positions]


