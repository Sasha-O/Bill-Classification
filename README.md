# Bill-Classification
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#1: Import modules needed
import requests 
import pandas as pd
import json
import numpy as np
import sklearn.cluster as cluster
import sklearn.model_selection as model_selection
import string
from string import punctuation
from sklearn.cluster import KMeans
#import nltk

#creates a function that will strip punctuation from a string 
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)


#creates a function that will strip numbers from a string 
numbers="0123456789"
def strip_numbers(s):
    return ''.join(c for c in s if c not in numbers)

#2: Use API to import the data
#set parameters
payload = {'key':'E8ycanKot3YTKvMQC2z0lBGNSQfoKwvi','full':'true','limit':'1000'}
#get bill data
bill = requests.get("http://legislation.nysenate.gov/api/3/bills/2017/",params=payload)
#convert bill data from API to string
bill_data = bill.text
#load bill_data as json
bill_data_json = json.loads(bill_data)

#3: Pull features from the data and organize/clean it
#Create list variables that will hold the information needed before moving to Pandas DataFrame
basePrintNo_list = []
Summary_list = []
#Looping over json data and store it into the list variable
for i in range(0,1000) :
    basePrintNo_list.append(str(bill_data_json['result']['items'][i]['basePrintNo']))
    Summary_list.append(str(bill_data_json['result']['items'][i]['summary']))
#Create  Panda DataFrame that is equal to the list variables appended to each other     
storage = pd.DataFrame()
storage[0] = basePrintNo_list
storage[1] = Summary_list
storage[2]=''

#Preprocess the bill summaries by:
#stripping punctuation and numbers away from the strings and copying those values into new column    
for i in range (0,1000):
    storage.iloc[i,2] = strip_punctuation(strip_numbers(storage.iloc[i,1]))

#append all the summaries into a single string variable and split it into a list
superstring = storage[2].to_string(header=False,index=False)
words=superstring.split()
#convert the standard list to a numpy array and remove all the duplicate words
words=np.array(words)
words_distinct=np.unique(words)
#set the number of features equal to the number of unique words 
features = len(words_distinct)
#append that many columns to storage pandas dataframe
features_df = pd.DataFrame(np.zeros((len(storage), features)))
result =pd.concat([storage,features_df],axis=1,ignore_index=True)

#turn binary feature value from 0 to 1 if summary contains that word in the word_distinct list
for r in range (0,len(result)):
    list_words_summary=result.iloc[r,2].split()
    #drop first 2 words, they are meaningless often 'establishes a...'
    list_words_summary=list_words_summary[2:]
    for c in range (3,len(result.columns)):
        if words_distinct[c-3] in list_words_summary:
            result.iloc[r,c]=1 

#Identify "empty bills" or bills without a summary we will remove these from the data set
count_without_summary=0
items_to_drop = []
for i in range(0,len(result)):
    if result.iloc[i,2]=='':
        count_without_summary=count_without_summary+1
        items_to_drop.append(i)
print(count_without_summary)
    #in my example 134 items didn't have a summary 

#remove the empty 'non-summary' bills and reset the index of the rows
result = result.drop(result.index[items_to_drop])
result = result.reset_index(drop=True)

#4: Visualize the data!
    #you can't really visualize non-2D or 3D data. Too many features, maybe try PCA? 


#5: Break data set out into 3 Sets: Training Set, Cross-Validation Set, and Test Set
# Split-out validation dataset
array = result.values
X = array[:,:] #Set feature variables
#Y = array[:,4] #Set output variables
validation_size = 0.20 #set the size of your validation set
seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_train, X_validation  = model_selection.train_test_split(X, test_size=validation_size, random_state=seed)

#Test K-Means Algorithm
kmeans = KMeans(n_clusters=50)
#make sure to omit non-feature columns
kmeans.fit(X_train[:,3:])
cluster_label = kmeans.labels_
clustered_X_train = np.c_[X_train,cluster_label]
clustered_X_train=pd.DataFrame(clustered_X_train)

#output result of cluster to csv        
finals.to_csv('initial_result.csv')

finals = clustered_X_train[[0,1,1724]]

'''
Initial takeaways!
definitley needs to exclude the first 2-3 words of the summary, they are always the same
special words dictionary would be very helpful
stemming still might be very helpful
certainly needs to have an exlusionary list like removing the of's fors etc.

'''


#Test several models both supervised and unsupervised
    #Keep in mind to use visualizations along the way to map bias and variance 
#Pick best model 
#Implement into online database 
