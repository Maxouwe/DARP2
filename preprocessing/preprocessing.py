import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from collections import Counter
import preprocessFunctions as pfs
import os
import descriptiveFunctions as dfs

##################         ##################
##################functions##################
##################         ################## 
def normalizeCSV(df, columnName):
    #remove punctuation
    pfs.removePunctuation(df,columnName)
    #turns each string under columnName into a list of strings
    pfs.tokenizeColumn(df, columnName)
    #remove stop words
    pfs.removeStopWordsFromColumn(df, columnName)
    #remove numbers and units
    pfs.removeUnitsAndNumbersFromColumn(df, columnName)   
    pfs.lemmatizeColumn(df, columnName)

def main():
    
    #dont forget to manually add product_descriptions.csv to the local directory if its not there
    pddf = pd.read_csv("resources/product_descriptions.csv", encoding="latin1")
    qpdf = pd.read_csv("resources/query_product.csv", encoding="latin1")

    #normalizing the product descriptions takes a long time
    #especially the lemmatization 
    #you have to wait until the console returns
    #normalize product descriptions
    if not os.path.exists("resources/normalized_pd.csv"):
        normalizeCSV(pddf, 'product_description')    
        pddf.to_csv("resources/normalized_pd.csv")

    #normalize queries and product title
    if not os.path.exists("resources/normalized_qp.csv"):
        normalizeCSV(qpdf, 'search_term')
        normalizeCSV(qpdf, 'product_title')     
        qpdf.to_csv("resources/normalized_qp.csv")

    



main()


