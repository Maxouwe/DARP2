import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import string

#not needed if youve executed these once already
#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('wordnet')

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

#lemmatizes words see google
def lemmatizeRow(row, columnName, lemmatizer):
    processed_sentence = []
    for w in row[columnName]:
        processed_sentence.append(lemmatizer.lemmatize(w))
    return processed_sentence

def lemmatizeColumn(df, columnName):
    lemmatizer = WordNetLemmatizer()
    df[columnName] = df.apply(lemmatizeRow, axis=1, args=[columnName, lemmatizer])

#make a small version product_description to speed up testing
def makeSmallProductDescriptionsTable(df):
    filtered_df = df[df['product_uid'] <= 100030]
    filtered_df.to_csv("resources/small_pd.csv")