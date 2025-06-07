import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from collections import Counter
import features

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
print(results.summary())

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