#KNN Classifier

#DELA PAZ, REYMAR
#RODRIGO, NATHANAEL JONAS SJ
#VERGEL DE DIOS, ANNE CATHERINE

#APRIL 1, 2019

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import itertools as it
from pandas import ExcelWriter
from pandas import ExcelFile


dataset=pd.read_csv('Iris_Data.csv')

columns=[0,1,2,3]
c4=list(it.combinations(columns,4))
c3=list(it.combinations(columns,3))
c2=list(it.combinations(columns,2))
c=c2+c3+c4

columns2=['sepal_length','sepal_width','petal_length','petal_width']
s4=list(it.combinations(columns2,4))
s3=list(it.combinations(columns2,3))
s2=list(it.combinations(columns2,2))
s=s2+s3+s4

classification_list=[]
kfold_list=[]
index_list=[]

for var in range (0, 11, 1):
    index_s=s[var]
    index=list(c[var])
    X = dataset.iloc[:,index].values
    Y = dataset.iloc[:,4].values
    print(index_s)
    
    dataset.isnull().sum().sort_values(ascending=False)
    dataset.dtypes
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
    
    from sklearn.preprocessing import StandardScaler
    StandardScaler = StandardScaler()
    X_train = StandardScaler.fit_transform(X_train)
    X_test = StandardScaler.fit_transform(X_test)
    
    from sklearn.neighbors import KNeighborsClassifier
    k_nearest_neighbors = KNeighborsClassifier(n_neighbors=33)
    k_nearest_neighbors.fit(X_train, Y_train)
    
    Y_predict = k_nearest_neighbors.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    classification_accuracy = accuracy_score(Y_test, Y_predict)
    print('Classification Accuracy: %.4f' %classification_accuracy)
    
    
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits = 10, random_state=0)
    
    from sklearn.model_selection import cross_val_score
    X_feature_scaled = StandardScaler.fit_transform(X)
    accuracies = cross_val_score(estimator = k_nearest_neighbors, X=X_feature_scaled, y = Y, cv = k_fold, scoring = 'accuracy') #X=X if no feature scaling
    accuracies_average = accuracies.mean()
    print('Cross - k-fold average = %.4f' %accuracies_average)
    print('')
    
    classification_list.append(classification_accuracy)
    kfold_list.append(accuracies_average)
    index_list.append(index_s)


#OUTPUT in EXCEL FILE
    
df = pd.DataFrame({'Features':index_list,
                   'Classification Accuracy':classification_list,
                   'Cross K-Fold Accuracy':kfold_list,})
writer = ExcelWriter('Output-KNN-Group1.xlsx')
df.to_excel(writer,'Sheet1',index=False)
writer.save()

# BEST COMBINATION = Features 'petal_length' and 'petal_width'
#Classification Accuracy: 0.7667
#Cross - k-fold average = 0.9400

