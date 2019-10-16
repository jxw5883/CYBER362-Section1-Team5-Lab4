# if you need to change directory or issue os-like commands

import os
os.getcwd()
#os.chdir('Data')
import pandas as pd

from sklearn import tree


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
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


tree_clf = tree.DecisionTreeClassifier()

tree_clf.fit(X_train, Y_train)
Y_predict = tree_clf.predict(X_test)

# import the metrics class and compute accuracy score
from sklearn import metrics
metrics.accuracy_score(Y_predict, Y_test)


cnf_matrix = metrics.confusion_matrix(Y_test, Y_predict)
print(cnf_matrix)

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
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#Calculate the F1 score
print(metrics.f1_score(Y_test, Y_predict))
#the classification report computes all the above.
print(metrics.classification_report(Y_test, Y_predict))
print(metrics.roc_curve(Y_test,Y_predict))
plt.show()

##visualize tree
# You need to install graphviz and pydotplus using the following commands
#pip install graphviz
#pip install pydotplus
#Then use following command to add the path of Graphviz
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin/'

from IPython.display import Image  
import pydotplus
from sklearn.tree import export_graphviz
# Create DOT data
dot_data = export_graphviz(tree_clf, out_file=None, 
                                feature_names=feature_cols,  
                               class_names=['0', '1'])

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data) 

# Show graph
Image(graph.create_png()) 

# in the above code for the visualizing the tree, you can also send output to a file and display from file
'''
from sklearn.externals.six import StringIO  
dot_data = StringIO()
dot_data = export_graphviz(tree_clf, out_file=dot_data, 
                                feature_names=feature_cols,  
                                class_names=['0', '1'])

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data.getvalues()) 
graph.write_png('dfraud.png')

# Show graph
Image(graph.create_png()) 
'''


             