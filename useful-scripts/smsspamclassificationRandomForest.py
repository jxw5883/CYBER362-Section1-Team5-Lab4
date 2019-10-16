# -*- coding: utf-8 -*-
"""

@author: jdk450
"""

#Created on Tue Sep  3 13:10:08 2019

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

# Let us try a RandomForest classifier

from sklearn.ensemble import RandomForestClassifier

#create a random forest classifier with 500 trees and a maximum of 20 items at leaf nodes
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=20)

rnd_clf.fit(X_train, y_train)
y_predict= rnd_clf.predict(X_test)

#get confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_predict)
print(cnf_matrix)

#Calculate performance measures:
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

#if you don't include pos_label='sham' you get this error:
#ValueError: pos_label=1 is not a valid label: array(['ham', 'spam'], dtype='<U4')
print("Precision:", metrics.precision_score(y_test, y_predict, pos_label='spam'))
print("Recall:",metrics.recall_score(y_test, y_predict, pos_label = 'spam'))
print("F1-score", metrics.f1_score(y_test, y_predict, pos_label='spam'))

print(metrics.classification_report(y_test, y_predict))

#want to plot ROC Curve but need the prediction probabilities for label using X_test for prediction
probs =rnd_clf.predict_proba(X_test)
preds=probs[:, 1]

#these are the y predictions in probability form
print(preds)

fpr, tpr, threshold = metrics.roc_curve(y_test, preds, pos_label='spam')
roc_auc = metrics.auc(fpr, tpr)

# Plot ROC curve - method 1
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--') #dash diagonal in red
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Plot ROC curve- method 2
def plot_roc_curve(fpr, tpr, label='spam'):
    plt.plot(fpr, tpr, linewidth=2, label='AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'b--') # dashed diagonal in blue
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
        
plot_roc_curve(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.show()






