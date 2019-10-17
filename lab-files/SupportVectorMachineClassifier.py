# -*- coding: utf-8 -*-
"""

@author: jxw5883
"""

import os
import emailReadUtility
import pandas  as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report

#make sure you know the directory where you are and locate trec07p in the appropriate directory
os.getcwd()
DATA_DIR = 'Data/trec07p/data/'
LABELS_FILE = 'Data/trec07p/full/index'
TESTING_SET_RATIO = 0.2

labels = {}
# Read the labels
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0
        
def read_email_files():
    X = []
    y = [] 
    for i in range(len(labels)):
        filename = 'inmail.' + str(i+1)
        email_str = emailReadUtility.extract_email_text(
            os.path.join(DATA_DIR, filename))
        X.append(email_str)
        y.append(labels[filename])
    return X, y

X, y = read_email_files()

#take a look at X and y . Look at the individual emails and index file to make sense of what you see.
pd.DataFrame(X).head()
pd.DataFrame(y).head()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

vectorizer = TfidfVectorizer()
X_train_vector= vectorizer.fit_transform(X_train)
X_test_vector= vectorizer.transform(X_test)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train_vector, y_train) 

Y_predict = svclassifier.predict(X_test_vector)

print(confusion_matrix(y_test, Y_predict))  
print(classification_report(y_test, Y_predict))
print("Precision:", precision_score(y_test, Y_predict))
print("Recall:", recall_score(y_test, Y_predict))
print(f1_score(y_test, Y_predict))
