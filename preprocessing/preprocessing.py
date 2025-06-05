import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from collections import Counter
import preprocessFunctions as pp
import os

#make a small version of product_descriptions to test functions faster
if os.path.exists("resources/small_pd.csv"):
    os.remove("resources/small_pd.csv")
df = pd.read_csv("resources/product_descriptions.csv", encoding="latin1")
pp.makeSmallProductDescriptionsTable(df)
df = pd.read_csv("resources/small_pd.csv", encoding="latin1")



#remove punctuation
pp.removePunctuation(df,'product_description')
#turns each string under columnName into a list of strings
pp.tokenizeColumn(df, 'product_description')
pp.lemmatizeColumn(df, 'product_description')
#remove stop words
pp.removeStopWordsFromColumn(df, 'product_description')
#remove numbers and units
pp.removeUnitsAndNumbersFromColumn(df, 'product_description')

#write to csv file
if os.path.exists("resources/processed_product_descriptions.csv"):
    os.remove("resources/processed_product_descriptions.csv")
df.to_csv("resources/processed_product_descriptions.csv")

