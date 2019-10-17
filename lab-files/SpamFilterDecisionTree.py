# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:51:44 2019

@author: jdk450
"""

import os
import emailReadUtility
import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report

#make sure you know the directory where you are and locate trec07p in the appropriate directory
os.getcwd()
DATA_DIR = 'trec07p/data/'
LABELS_FILE = 'trec07p/full/index'
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

X_train, X_test, y_train, y_test, idx_train, idx_test = \
    train_test_split(X, y, range(len(y)), 
    train_size=TESTING_SET_RATIO, random_state=2)

vectorizer = TfidfVectorizer()
X_train_vector= vectorizer.fit_transform(X_train)
X_test_vector= vectorizer.transform(X_test)


# Initialize the classifier and make label predictions
#cl_lr = LogisticRegression()
#cl_lr.fit(X_train_vector, y_train)
tree = tree.DecisionTreeClassifier()
tree.fit(X_train_vector, y_train)
#y_pred = cl_lr.predict(X_test_vector)
y_pred = tree.predict(X_test_vector)

#get confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" , fmt='g'); plt.title('Confusion matrix') ; plt.ylabel('Actual label'); plt.xlabel('Predicted label')


#show confusion matrix
print(cnf_matrix)

# compute and Print performance metrics
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_pred)))

#if you don't include pos_label='sham' you get this error:
#ValueError: pos_label=1 is not a valid label: array(['ham', 'spam'], dtype='<U4')
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

