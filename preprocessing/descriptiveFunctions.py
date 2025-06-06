import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from collections import Counter
import preprocessFunctions as pf
import os
import descriptiveFunctions as df

def showRelevanceScoreDistribution(df):
    plt.hist(df['relevance'])
    plt.show()
    
def showCountUniqueQueries(df):
    print(df['search_term'].nunique())


    
    

