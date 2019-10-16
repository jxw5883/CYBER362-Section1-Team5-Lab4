# -*- coding: utf-8 -*-
"""

# -*- coding: utf-8 -*-
Created on Tue Sep  3 13:10:08 2019

@author: jdk450
"""

import os
import pandas  as pd
from sklearn import model_selection, linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.getcwd()
#os.chdir('Python') # this depends on where you placed the dataset

# load the dataset
data = pd.read_csv('Data/SMSSpamCollection.csv')

# create  dataframes using texts and labels
texts = data.iloc[:, 1]
labels = data.iloc[:, 0]

#take a look
texts
labels

# split the dataset into training and validation datasets 
X_train, X_test, y_train, y_test = train_test_split(texts,labels, test_size=0.20,random_state=0)
vectorizer = TfidfVectorizer()
X_train= vectorizer.fit_transform(X_train)
X_test= vectorizer.transform(X_test)

# Let us try some voting classifiers

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#create three classifiers and a voting classifier
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(C=1.0, kernel='linear')
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
voting_clf.fit(X_train, y_train)

#Let’s look at each classifier’s accuracy on the test set:
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(clf.__class__.__name__, ", Confusion Matrix:\n", cnf_matrix)
    print(clf.__class__.__name__, ", Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("F1-score", metrics.f1_score(y_test, y_pred, pos_label='spam'))

