# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:51:44 2019

@author: jdk450
"""

import os
import emailReadUtility
import pandas  as pd
from sklearn import model_selection, linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns


#make sure you know the directory where you are and locate trec07p in the appropriate directory
os.getcwd()
DATA_DIR = '/Users/jaketompkins/Downloads/trec07p/data'
LABELS_FILE = '/Users/jaketompkins/Downloads/trec07p/full/index'
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
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=20)
rnd_clf.fit(X_train_vector, y_train)
y_pred = rnd_clf.predict(X_test_vector)

#get confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

#show confusion matrix
print(cnf_matrix)

# compute and Print performance metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#if you don't include pos_label='sham' you get this error:
#ValueError: pos_label=1 is not a valid label: array(['ham', 'spam'], dtype='<U4')
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F-1Score:", metrics.f1_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))

probs =rnd_clf.predict_proba(X_test)
preds=probs[:, 1]

print(preds)

fpr, tpr, threshold = metrics.roc_curve(y_test, preds, pos_label='spam')
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

def plot_roc_curve(fpr, tpr, label='spam'):
    plt.plot(fpr, tpr, linewidth=2, label='AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
plot_roc_curve(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.show()
