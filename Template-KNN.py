# Template for KNN Classifier 
# April 1, 2019

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

dataset_statistics = dataset.iloc[:,1:]
print(dataset_statistics.describe())

X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

dataset.isnull().sum().sort_values(ascending=False)
dataset.dtypes

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=0)

from sklearn.preprocessing import StandardScaler
standard_scaler_X = StandardScaler() 
X_train = standard_scaler_X.fit_transform(X_train) 
X_test = standard_scaler_X.fit_transform(X_test) 

# ============== MACHINE LEARNING ==============

# TO FIT THE TRAINING DATASET INTO K-NEAREST NEIGHBORS MODEL
from sklearn.neighbors import KNeighborsClassifier
k_nearest_neighbors = KNeighborsClassifier(n_neighbors=33)
k_nearest_neighbors.fit(X_train,Y_train) 

# TO PREDICT OUTPUT OF THE TESTING DATASET
Y_predict = k_nearest_neighbors.predict(X_test)

# SHOW CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict) #column of confusion matrix is predicted values

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict)
print('Classification Accuracy: %.4f'
      %classification_accuracy)
print('')

classification_error = 1 - classification_accuracy 
print('Classification Error: %.4f'
      %classification_error)
print('')

from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test,Y_predict)
print('Sensitivity: %.4f'
      %sensitivity)
print('')

specificity = TN/(TN+FP)
print('Specificity: %.4f'
      %specificity)
print('')

FPR = FP/(TN+FP)
print('False Postive Rate: %.4f'
      %FPR)
print('')

from sklearn.metrics import precision_score
precision = precision_score(Y_test,Y_predict)
print('Precision: %.4f'
      %precision)
print('')

from sklearn.metrics import f1_score
f1_score = f1_score(Y_test,Y_predict)
print('F1 Score: %.4f'
      %f1_score)
print('')

from sklearn.metrics import classification_report
classification_report = classification_report(Y_test,Y_predict)

# TO APPLY K-FOLD CROSS VALIDATION FOR THE MODEL'S PERFORMANCE
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, random_state=0)

# to validate
from sklearn.model_selection import cross_val_score
X_feature_scaled = standard_scaler_X.fit_transform(X) 
accuracies = cross_val_score(estimator = k_nearest_neighbors, X = X_feature_scaled, y = Y, cv = k_fold, scoring = 'accuracy') # X = X if no feature scaling
accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

# TO VISUALIZE THE TRAINING DATASET RESULTS (HOLD OUT) 
from matplotlib.colors import ListedColormap
X_set_train, Y_set_train = X_train, Y_train
X1_train, X2_train = np.meshgrid(np.arange(start = X_set_train[:, 0].min() - 1, stop = X_set_train[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set_train[:, 1].min() - 1, stop = X_set_train[:, 1].max() + 1, step = 0.01))
plot.contourf(X1_train, X2_train, k_nearest_neighbors.predict(np.array([X1_train.ravel(), X2_train.ravel()]).T).reshape(X1_train.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plot.xlim(X1_train.min(), X1_train.max())
plot.ylim(X2_train.min(), X2_train.max())
for i, j in enumerate(np.unique(Y_set_train)):
    plot.scatter(X_set_train[Y_set_train == j, 0], X_set_train[Y_set_train == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plot.title('K-Nearest Neighbor (Training Dataset)')
plot.xlabel('PARAMETR 2')
plot.ylabel('PARAMETER 1')
plot.legend()
plot.show()

# TO VISUALIZE THE TESTING DATASET RESULTS (HOLD OUT)
from matplotlib.colors import ListedColormap
X_set_test, Y_set_test = X_test, Y_test
X1_test, X2_test = np.meshgrid(np.arange(start = X_set_test[:, 0].min() - 1, stop = X_set_test[:, 0].max() + 1, step = 0.01),
                   np.arange(start = X_set_test[:, 1].min() - 1, stop = X_set_test[:, 1].max() + 1, step = 0.01))
plot.contourf(X1_test, X2_test, k_nearest_neighbors.predict(np.array([X1_test.ravel(), X2_test.ravel()]).T).reshape(X1_test.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plot.xlim(X1_test.min(), X1_test.max())
plot.ylim(X2_test.min(), X2_test.max())
for i, j in enumerate(np.unique(Y_set_test)):
    plot.scatter(X_set_test[Y_set_test == j, 0], X_set_test[Y_set_test == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plot.title('K-Nearest Neighbor  (Testing Dataset)')
plot.xlabel('PARAMETR 2')
plot.ylabel('PARAMETER 1')
plot.legend()
plot.show()
