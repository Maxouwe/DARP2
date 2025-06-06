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
def tokenizeProductDescription(df):
    #remove punctuation
    pfs.removePunctuation(df,'product_description')
    #turns each string under columnName into a list of strings
    pfs.tokenizeColumn(df, 'product_description')
    pfs.lemmatizeColumn(df, 'product_description')
    #remove stop words
    pfs.removeStopWordsFromColumn(df, 'product_description')
    #remove numbers and units
    pfs.removeUnitsAndNumbersFromColumn(df, 'product_description')   

def main():
    #make a small version of product_descriptions to test functions faster
    # if os.path.exists("resources/small_pd.csv"):
    #     os.remove("resources/small_pd.csv")

    # df = pd.read_csv("resources/product_descriptions.csv", encoding="latin1")
    # pp.makeSmallProductDescriptionsTable(df)
    # df = pd.read_csv("resources/small_pd.csv", encoding="latin1")

    # tokenizeProductDescription(df)

    # #write to csv file
    # if os.path.exists("resources/processed_product_descriptions.csv"):
    #     os.remove("resources/processed_product_descriptions.csv")
    # df.to_csv("resources/processed_product_descriptions.csv")
    # Specify the path to your CSV file
    csv_path = "resources/query_product.csv"

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path, encoding="latin1")
    dfs.showCountUniqueQueries(df)



main()


