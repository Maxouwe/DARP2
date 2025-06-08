import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import string
from nltk.corpus import wordnet
from nltk.tag import pos_tag

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