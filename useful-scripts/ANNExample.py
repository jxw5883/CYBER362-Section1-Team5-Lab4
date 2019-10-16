# if you need to change directory or issue os-like commands
import os
os.getcwd()
os.chdir('Data/')
import pandas as pd
#col_names = ['accountAgeDays', 'numItems', 'localTime', 'paymentMethod', 'paymentMethodAgeDays', 'label'], we already have labels, so not needed
# load dataset and take a look
dfraud = pd.read_csv("payment_fraud.csv")
dfraud.head()
dfraud.sample()

#paymentMethod is categorical and will change it to nominal values
dfraud['paymentMethod'] = dfraud['paymentMethod'].map({'creditcard': 1, 'paypal': 2, 'storecredit': 3})

#take a look at the data now
dfraud.head()
# X=dfraud.iloc[:,0:5]
# Y = dfraud.iloc[:,5]
#split dataset in features and target variable
feature_cols = ['accountAgeDays', 'numItems', 'localTime', 'paymentMethod', 'paymentMethodAgeDays'] 
X = dfraud[feature_cols] # Features
Y = dfraud.label # Target variable

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

# import the class and create classifier/model
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(30, 30, 20),activation='relu',random_state=101)

#train model
model.fit(X_train, Y_train)  

#predict
Y_predict = model.predict(X_test)

# import confusion_matrix and classification_report classes
from sklearn.metrics import classification_report, confusion_matrix  

#computer performance measures
print(confusion_matrix(Y_test, Y_predict))  
print(classification_report(Y_test, Y_predict))  

from sklearn import metrics
#Calculate performance measures:
print("Accuracy:",metrics.accuracy_score(Y_test, Y_predict))
print("Precision:",metrics.precision_score(Y_test, Y_predict))
print("Recall:",metrics.recall_score(Y_test, Y_predict))

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
cnf_matrix =confusion_matrix(Y_test, Y_predict)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

