def check_words(row):
    words_column1 = set(row['product_title'].lower().split())
    words_column2 = set(row['search_term'].lower().split())
    return int(words_column2.issubset(words_column1))

def getPredictedRelevance(row):
    probs = [row['p1'], row['p2'], row['p3']]
    max = 0
    classification = -1
    for i in [0, 1, 2]:
        if probs[i] > max:
            max = probs[i]
            classification = i + 1
    return classification


    

