import numpy as np #Numerical Python
import matplotlib.pyplot as plot 
import pandas as pd #Panel Data

#2. IMPORT THE DATASET
dataset=pd.read_csv("/Users/annecatherinevergeldedios/Desktop/5Y2S/Machine Learning/Machine Learning 2019/Database/Salary_Data.csv")

#to create the matrix of independent variable, X
X=dataset.iloc[:,0:1].values # integer-location based indexing
#X=dataset.iloc[:,0:-1] 

#to create the matrix of dependent variable, Y
Y=dataset.iloc[:,1:2].values
#Y=dataset.iloc[:,1].values

#4. to handle the missing data

#to know how much of the data is missing
dataset.isnull().sum().sort_values(ascending=False)

# [ NO NEED TO IMPUTE ]

# 5. to split the whole dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# [ NO NEED TO USE FEATURE SCALING bc there is only 1 feature ]

# 7. TO fit the training dataset into a simple linear regression
from sklearn.linear_model import LinearRegression
linear_regression=LinearRegression() #to change class to object & model
linear_regression.fit(X_train,Y_train)

# 8. TO PREDICT OUTPUT OF THE TESTING DATASET
Y_predict=linear_regression.predict(X_test)

# 9. TO VISUALIZE THE TRAINING DATASET AND THE SIMPLE LINEAR REGRESSION MODEL
plot.scatter(X_train,Y_train,color='red')
Y_predict_Xtrain=linear_regression.predict(X_train) #to overlay the plot with the point and for the Y_predict
plot.plot(X_train,Y_predict_Xtrain,color='blue') #plot of linear regression model
plot.title('Plot of years of experience VS Salary using the training dataset')
plot.xlabel('Years of experience',color='red')
plot.ylabel('Salary',color='red')

#to create legend
import matplotlib.patches as mpatches
red_patch=mpatches.Patch(color='red', label='training dataset')
blue_patch=mpatches.Patch(color='blue', label='simple linear model')
plot.legend(handles=[red_patch,blue_patch])
plot.show()

# 10. TO VISUALIZE THE TESTING DATASET AND THE SIMPLE LINEAR REGRESSION MODEL
plot.scatter(X_test,Y_test,color='green')
plot.plot(X_train,Y_predict_Xtrain,color='blue') #plot of linear regression model
plot.title('Plot of years of experience VS Salary using the testing dataset')
plot.xlabel('Years of experience',color='yellow')
plot.ylabel('Salary',color='red')

#to create legend
green_patch=mpatches.Patch(color='green', label='test dataset')
blue_patch=mpatches.Patch(color='blue', label='simple linear model')
plot.legend(handles=[green_patch,blue_patch])
plot.show()


# 11. TO VISUALIZE THE PREDICTED SALARY (Y_predict) and the simple linear regression model
plot.scatter(X_test,Y_predict,color='yellow') #for the plot of the predicted salary
plot.plot(X_train,Y_predict_Xtrain,color='blue') #plot of linear regression model
plot.title('Plot of years of experience VS Salary using the training dataset')
plot.xlabel('Years of experience',color='yellow')
plot.ylabel('Salary',color='red')

#to create legend
yellow_patch=mpatches.Patch(color='yellow', label='Predicted Salary')
blue_patch=mpatches.Patch(color='blue', label='simple linear model')
plot.legend(handles=[yellow_patch,blue_patch])
plot.show()

# 11. TO VISUALIZE THE ACTUAL SALARY (Y_test) and the predicted salary (Y_predict)
plot.scatter(X_test,Y_test,color='green') #for the plot of the actual salary
plot.scatter(X_test,Y_predict,color='yellow') #plot of the predicted salary
plot.title('Plot of Actual Salary VS Predicted Salary')
plot.xlabel('Years of experience',color='yellow')
plot.ylabel('Salary',color='red')

#to create legend
yellow_patch=mpatches.Patch(color='yellow', label='Predicted Salary')
green_patch=mpatches.Patch(color='green', label='Actual Salary')
plot.legend(handles=[yellow_patch,green_patch])
plot.show()

# 12. TO DETERMINE THE INTERCEPT AND COEFFICIENT OF THE SIMPLE LINEAR REGRESSION
# A. FOR THE INTERCEPT AND COEFFICIENT

intercept=linear_regression.intercept_
coefficient=linear_regression.coef_ 

print('Intercept= %.2f'%intercept)
print('Coefficient= %.2f'%coefficient)

# model = 26780.10+9312.58*(Years of Experience)


# 13. EVALUATE THE PREFORMANCE OF THE SIMPLE LINEAR REGRESSION MODEL

#A. MEAN ABSOLUTE ERROR (MSE) - MEASURES HOW CLOSE THE FORECAST TO THE ACTUAL 
from sklearn.metrics import mean_absolute_error
MAE=mean_absolute_error(Y_test,Y_predict)
print('MEAN ABSOLUTE ERROR= %.4f'%MAE)

#B. MEAN SQUARED ERROR (MSE) 
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(Y_test,Y_predict)
print('MEAN SQUARED ERROR= %.4f'%MSE)

#C. ROOT MEAN SQUARED ERROR/DEVIATION (RMSE / RMSD) 
from math import sqrt
RMSE=sqrt(MSE)
print('ROOT MEAN SQUARED ERROR= %.4f'%RMSE)

#D. EXPLAINED VARIANCE SCORE (EVS)
from sklearn.metrics import explained_variance_score
EVS=explained_variance_score(Y_test,Y_predict)
print('EXPLAINED VARIANCE SCORE= %.4f'%EVS)

#E. COEFFICIENT OF DETERMINATION REGRESSION SCORE FUNCTION (R^2)- PERCENTAGE ON HOW ...
from sklearn.metrics import r2_score
R2=r2_score(Y_test,Y_predict)
print('R^2 SCORE= %.4f'%R2)

