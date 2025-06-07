import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import features
from statsmodels.miscmodels.ordinal_model import OrderedModel


#There are two sections here 
#First section shows how to do linear regression
#Second section shows how to do ordinal logistic regression

# Specify the path to your CSV file
csv_path = "resources/query_product.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_path, encoding="latin1")

# Set the training size
training_size = 50000

# Split the DataFrame into training and test sets
train_data, test_data = train_test_split(df, test_size=(len(df) - training_size), random_state=42)

# Creates a tables containing product_query_id and a score for all_words_in_title 
train_data['all_words_in_title'] = train_data.apply(features.check_words, axis=1)
test_data['all_words_in_title'] = test_data.apply(features.check_words, axis=1)

# Define the feature and target variables
X = train_data['all_words_in_title']
y = train_data['relevance']

# Add a constant term to the feature variable
# This is so the model will also create an intercept, that is coefficient B0
X = sm.add_constant(X)

# Create a linear regression model object
model = sm.OLS(y, X)

# Fit the model to the data, i.e. calculate the coefficients etc. see console output
# P>|t| value basically means, what is the chance the relevance does not depend on your feature
# If its smaller than 0.01 then its a significant feature
# If not then we should adjust/make a new feature
# Here you can see all_words_in_title is a significant feature
results = model.fit()
# print(results.summary())

#results now contain the model for the relevance score based on your features
#now you just have to inject the test data into the model
#and compare output of the model with the actual relevance of the test data

X_test = test_data['all_words_in_title']
y_test = test_data['relevance']

#so the B0 coefficient also gets taken into the calculation
#when you inject a test_data tuple into the model function B0 + B1X1 + B2X2 ... BnXn
#you multiply every coefficient Bi with the corresponding vector entry Xi
#the model expects B0 also have a vector entry to be multiplied with
#so we just give it a 1
X_test = sm.add_constant(X_test)

#generate try to predict relevance of test data
#i.e. put every tuple into the model function and see what the model outputs as relevance score
y_pred = results.predict(X_test)

#compare it with the actual relevance of the test data
#r-squared ranges from 0 to 1, from bad to good
#the r2-score is 0.05 which is very low
#Doesnt mean that all words in title is a bad feature
# Because we saw from its P>|t| value that its a good feature
# This just means our model is incomplete because it cant just rely on one feature 
r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)

#Counter is a dictionary d
#this one contains for each value yi in y_test d[yi] = amount of occurrences of yi in y_test
weight_counter = Counter(y_test)
#so the size of the point for value yi in the plot corresponds with how many times it occurs in the data
weights = [weight_counter[i]/10 for i in y_test]

#x-axis is score for all_words_in_title
#y-axis is the relevance 
#make scatter plot
plt.scatter(test_data['all_words_in_title'], y_test, label='Actual', s=weights)

#also make a line for how the model predicts relevance score based on the feature
#you can see the model predicts if all_words_in_title = 1 then relevance score is higher
plt.plot(test_data['all_words_in_title'], y_pred, color='red', label='Fitted Line')

plt.xlabel('Do all query terms occur in the product title?')
plt.ylabel('Relevance Score')
plt.title('Linear Regression: Fitted Line')
plt.legend()

#all the previous plt.f() calls modify some plot object
#call plt.show() to actually show the plot
plt.show()

##########################################################################################################
#so that concludes the linear regression part now there will be an example for ordinal logistic regression
#you have to close the last plot so the program can to continue i think
##########################################################################################################

#create logistic train and test data
#.where() is basically like WHERE in SQL
#here we want to only get the tuples where relevance is an integer
#the weird thing about .where() is that it still returns all tuples
#but for all tuples that do not meet the where-requirement the column values are replaced with NaN
#use .dropna() to remove these NaN tuples
logit_train_data = train_data.where(train_data.relevance == np.floor(train_data.relevance)).dropna()
logit_test_data = test_data.where(test_data.relevance == np.floor(test_data.relevance)).dropna()

#the all_words_in_title feature was already applied to the data in the linear regression section
#ordinal logistic regression does not use an intercept so we dont add a column of 1's to X like we did in linear regression
X = logit_train_data['all_words_in_title']
y = logit_train_data['relevance']


#create ordinal logistic model object
logit_model = OrderedModel(y, X, distr='logit')

#calculate coefficients and thresholds
#het model berekent een waarde bepaalde waard x 
#en binnen in het model wordt deze waarde x gebruikt om de input the classificeren
#Dus deze waarde x zie je als gebruiker niet
#in de summary zie je 3 coefficienten een voor de allwordsintitle feature
#en twee thresholds
#als de onzichtbare waarde x < threshold 1.0/2.0 dan is de output relevance = 1
#als x tussen de twee thresholds ligt is output relevance = 2
#is x groter dan threshold 2.0/3.0 dan is de output relevance = 3
#ook hier kun je aflezen met P>|Z| hoe significant de coefficienten zijn (net zoals bij linear regression)
#het blijkt dat alle coefficienten hier significant zijn
#dus ook voor ordinal logistic regression is all_words_in_title een goede feature
logit_results = logit_model.fit()
# print(logit_results.summary())

#now we are going to test the model on the test data
X_test = logit_test_data['all_words_in_title']
y_test = logit_test_data['relevance']

#see how the model classifies the test_data tuples
#the linear regression predict() function can take the data directly
#but for the logistic regression predict() you for some reason first need to cast 
#the data to an numPy.Array()
#the output of OrderedModel.predict() 
#for each tuple x we get output [p1, p2, p3]
#where p1 is the chance that x has relevance score 1, p2 to relevance score 2, p3 to relevance score 3
y_pred = logit_results.predict(np.array(X_test))

#so to extract the predicted class we have to choose the highest probability from the array for each tuple
#the annoying thing is that OrderedModel.predict() returns a different datatype than OLS.predict()
#so we have to reformat things to make it work
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['p1', 'p2', 'p3']
y_pred['predicted_relevance'] = y_pred.apply(features.getPredictedRelevance, axis=1)
y_pred.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

#i reset the indices for both tables
#so now both tables indices go from 0 to size(test_data)-1
#then joined them by index, which is now because of the index reset a row-wise join
#if logit_results.predict() does not reorder the rows
#and if reset_index does not reorder the rows 
#this should be good
#you can see in the console that p3 is the highest for every tuple
#because our model is one sided because it has only one feature
#so our model predicts every tuple to be of relevance level 3
print(pd.concat([y_test, y_pred], axis=1)[['relevance','p1', 'p2', 'p3', 'predicted_relevance']])

#you can read the confusion matrix as follows
#on the bottom you read predicted class 0 to 2 (which is relevance level 1 to 3)
#and for each predicted class you look up vertically
#we have predicted all tuples to be in class 3
#you look in the class 3 column to see how many of your class 3 predictions
#were in reality in class1, class2, class3
#we can see that if we predict class 3 then not alot are actually in class 1
#but quite alot of them are actually in class 2
#but the good thing is that most of our class 3 predictions were actually correct
cm = confusion_matrix(y_test, y_pred['predicted_relevance'])
disp = ConfusionMatrixDisplay(confusion_matrix= cm)
disp.plot()
plt.show()