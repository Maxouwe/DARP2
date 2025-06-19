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
from preprocessFunctions import add_vector_similarities
import gensim.downloader as gensim_api
from ast import literal_eval

# Only needs to be executed once
w2v_model = gensim_api.load('word2vec-google-news-300')

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

    # pddf = pd.read_csv("resources/normalized_pd.csv", encoding="latin1")
    # pfs.turnStringFieldToList(pddf, 'normalized_pd')


    #normalize queries and product title
    if not os.path.exists("resources/normalized_qp.csv"):
        qpdf = pd.read_csv("resources/normalized_qp.csv", encoding="latin1")
        normalizeCSV(qpdf, 'search_term')
        normalizeCSV(qpdf, 'product_title')     
        qpdf.to_csv("resources/normalized_qp.csv") 
        
    # qpdf = pd.read_csv("resources/normalized_qp.csv", encoding="latin1")
    # pfs.turnStringFieldToList(qpdf, 'normalized_st')
    # pfs.turnStringFieldToList(qpdf, 'normalized_title')
    

    #make qfscore table
    if not os.path.exists("resources/qf_scores.csv"):
        qfScores = pfs.createQFScores(qpdf)
        qfScores.to_csv("resources/qf_scores.csv")
    #idfscore table
    if not os.path.exists("resources/qp_idf_scores.csv"):
        qp_idf_Scores = pfs.createIDFScores(qpdf.drop_duplicates(subset=['product_uid']), 'normalized_title')
        qp_idf_Scores.to_csv("resources/qp_idf_scores.csv")
        
    #for proximity score feature
    if not os.path.exists("resources/word_positions.csv"):
        print('making word_posts')
        pos_lists = pfs.createPosLists(pddf)
        pos_lists.to_csv("resources/word_positions.csv", index=False)

    # pos_lists = pd.read_csv("resources/word_positions.csv")
    # pfs.turnStringFieldToList(pos_lists, 'position_lists')

    #make product description idf table
    #do not run this, it takes for ever,
    #get pd_idf_scores.csv from my branch
    # if not os.path.exists("resources/pd_idf_scores.csv"):
    #     pdIDFScores = pfs.createIDFScores(pddf, 'normalized_pd')
    #     pdIDFScores.to_csv("resources/pd_idf_scores.csv")

    if not os.path.exists("resources/term_freqs.csv"):
        # Specify the path to your CSV file
        csv_path = "resources/query_product.csv"

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path, encoding="latin1")
        
        nqpdf = pd.read_csv("resources/normalized_qp.csv")
        npddf = pd.read_csv("resources/normalized_pd.csv")
        df = df.join(nqpdf.set_index('id'), on='id')
        df = df.join(npddf.set_index('product_uid'), on='product_uid')
        df['normalized_st'] = df['normalized_st'].apply(literal_eval)
        df['normalized_title'] = df['normalized_title'].apply(literal_eval)
        df['normalized_pd'] = df['normalized_pd'].apply(literal_eval)
        df['term_freqs'] = df.apply(pfs.getTermFrequencies, axis=1)
        df = df.drop('product_uid', axis=1)
        df = df.drop('search_term', axis=1)
        df = df.drop('relevance', axis=1)
        df = df.drop('normalized_title', axis=1)
        df = df.drop('normalized_st', axis=1)
        df = df.drop('normalized_pd', axis=1)
        df = df.drop('product_title', axis=1)
        df.to_csv("resources/term_freqs.csv", index=False)
    
    if not os.path.exists("resources/tfidf_score.csv"):
        df = pd.read_csv("resources/query_product.csv", encoding="latin1")
        print("done reading1")
        df = df.drop('product_title', axis=1)
        df = df.drop('search_term', axis=1)
        df = df.drop('relevance', axis=1)


        nqpdf = pd.read_csv("resources/normalized_qp.csv")
        print("done reading 2")
        nqpdf = nqpdf.drop('normalized_title', axis=1)
        

        npddf = pd.read_csv("resources/normalized_pd.csv")
        tfdf = pd.read_csv("resources/term_freqs.csv")
        print("done reading 2")


        df = pd.merge(df, tfdf, on='id', how='inner')
        df = pd.merge(df, nqpdf, on='id', how='inner')
        df = pd.merge(df, npddf, on='product_uid', how='inner')
        
        print("done merging")

        df = df.drop('product_uid', axis=1)
        
        df['normalized_st'] = df['normalized_st'].apply(literal_eval)
        df['normalized_pd'] = df['normalized_pd'].apply(literal_eval)
        df['term_freqs'] = df['term_freqs'].apply(literal_eval)

        idfdf = pd.read_csv("resources/pd_idf_scores.csv")
        print("start applying")
        df['tfidf'] = df.apply(pfs.getTFIDFScore, axis=1, args=[idfdf])
        
        df = df.drop('normalized_st', axis=1)
        df = df.drop('normalized_pd', axis=1)
        df = df.drop('term_freqs', axis=1)

        print("IO")

        df.to_csv("resources/tfidf_score.csv", index=False)

    #word embedding
    if not os.path.exists("resources/qp_with_vecsim.csv"):
        # Specify the path to your CSV file
        csv_path = "resources/query_product.csv"

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path, encoding="latin1")
        
        nqpdf = pd.read_csv("resources/normalized_qp.csv")
        npddf = pd.read_csv("resources/normalized_pd.csv")
        df = df.join(nqpdf.set_index('id'), on='id')
        df = df.join(npddf.set_index('product_uid'), on='product_uid')
        df['normalized_st'] = df['normalized_st'].apply(literal_eval)
        df['normalized_title'] = df['normalized_title'].apply(literal_eval)
        df['normalized_pd'] = df['normalized_pd'].apply(literal_eval)

        # Voeg samen:
        #ik kreeg hier een merge error maar weet niet waarom
        #dus ik heb ff mijn manier van mergen gedaan bovenaan in main
        # merged = qpdf.merge(pddf[['product_uid', 'normalized_pd']], on='product_uid', how='left')
        merged = df

        #alles mergen in één grote tabel hoeft pas in main
        # merged = merged.merge(qfScores, how='cross')
        # merged = merged.merge(qp_idf_Scores, how='cross')
        # merged = merged.merge(pd_idf_Scores, how='cross')
        
        # Voeg vector features toe
        merged = add_vector_similarities(merged, w2v_model)
        
        #verwijder overbodige kolommen
        merged = merged.drop('product_uid', axis=1)
        merged = merged.drop('search_term', axis=1)
        merged = merged.drop('relevance', axis=1)
        merged = merged.drop('normalized_title', axis=1)
        merged = merged.drop('normalized_st', axis=1)
        merged = merged.drop('normalized_pd', axis=1)
        merged = merged.drop('product_title', axis=1)
        # Bewaar alles:
        merged.to_csv("resources/qp_with_vecsim.csv", index=False)

main()


