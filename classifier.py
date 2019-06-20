import pandas as pd
import numpy as np

# read data
data = pd.read_csv('datasetv1.csv')

# normalise column Milliseconds
minMil = data['Milliseconds'].min()
maxMil = data['Milliseconds'].max()
data['Milliseconds'] = (data['Milliseconds'] - minMil)/(maxMil - minMil)

data.fillna(0, inplace = True)
# seperate prediction value
labels = np.array(data['Prediction'])

# prepare data for training
data = data.drop('Prediction', axis=1)
# one hot encoding rules
data = pd.get_dummies(data)

features_list = list(data.columns)
data = np.array(data)




# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 42)

# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)


# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

accuracy = accuracy_score(test_labels, predictions)
presicion = precision_score(test_labels, predictions, average = 'weighted')
recall = recall_score(test_labels, predictions, average='weighted')

print('Accuracy:', accuracy, '%.')
print('Precision:', presicion, '%.')
print('Recall:', recall, '%.')


estimator = rf.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = features_list,
                class_names = ['0', '1', '2'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')