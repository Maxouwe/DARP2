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
    
    #normalizing the product descriptions takes a long time
    #especially the lemmatization 
    #you have to wait until the console returns
    #normalize product descriptions
    if not os.path.exists("resources/normalized_pd.csv"):
        pddf = pd.read_csv("resources/product_descriptions.csv", encoding="latin1")
        normalizeCSV(pddf, 'product_description')    
        pddf.to_csv("resources/normalized_pd.csv")

    pddf = pd.read_csv("resources/normalized_pd.csv", encoding="latin1")
    pfs.turnStringFieldToList(pddf, 'product_description')

    #normalize queries and product title
    if not os.path.exists("resources/normalized_qp.csv"):
        qpdf = pd.read_csv("resources/normalized_qp.csv", encoding="latin1")
        normalizeCSV(qpdf, 'search_term')
        normalizeCSV(qpdf, 'product_title')     
        qpdf.to_csv("resources/normalized_qp.csv") 
        
    qpdf = pd.read_csv("resources/normalized_qp.csv", encoding="latin1")
    pfs.turnStringFieldToList(qpdf, 'search_term')
    pfs.turnStringFieldToList(qpdf, 'product_title')

    #make qfscore table
    if not os.path.exists("resources/qf_scores.csv"):
        qfScores = pfs.createQFScores(qpdf)
        qfScores.to_csv("resources/qf_scores.csv")
    
    if not os.path.exists("resources/qp_idf_scores.csv"):
        qp_idf_Scores = pfs.createIDFScores(qpdf.drop_duplicates(subset=['product_uid']), 'product_title')
        qp_idf_Scores.to_csv("resources/qp_idf_scores.csv")

    #make product description idf table
    #do not run this, it takes for ever,
    #get pd_idf_scores.csv from my branch
    # if not os.path.exists("resources/pd_idf_scores.csv"):
    #     pdIDFScores = pfs.createIDFScores(pddf, 'product_description')
    #     pdIDFScores.to_csv("resources/pd_idf_scores.csv")



main()


