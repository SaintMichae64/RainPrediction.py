import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
import requests

# Downloading and loading the Dataset into Pandas DataFrame
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-Final

url = path
response = requests.get(url)

with open("Weather_Data.csv", "wb") as file:
    file.write(response.content)

data = pd.read_csv('Weather_Data.csv')
data.head()

# Data Preprocessing, convert categorical variables to binary variables.

data_processed = pd.get_dummies(data=data, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
data_processed.replace(['Yes', 'No'], [1,0], inplace=True)
data_processed.drop('Date', axis=1, inplace=True)
data_processed = data_processed.astype(float)
X = data_processed.drop(columns='RainTomorrow', axis=1)
y = data_processed['RainTomorrow']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=10)
print ('Training set:', X_train.shape,  y_train.shape)
print ('Testing set:', X_test.shape,  y_test.shape)

# Using Linear Regression
LinearRegr = LinearRegression()
X = X_train.values
y = y_train.values
LinearRegr.fit(X, y)

# The coefficients and intercept
print('Coefficient: ', LinearRegr.coef_[0:5])
print('Intercept: ', LinearRegr.intercept_)

# Prediction and Evaluation - this throws an error and needs fixed  (C:\Users\wyzzhi fabulouss\Videos\Anaconda3\lib\site-packages\sklearn\base.py:413: UserWarning: X has feature names, but LinearRegression was fitted without feature names
  warnings.warn)

prediction = LinearRegr.predict(X_test.values)
X = X_test.values
y = y_test.values
print("Residual sum of squares: %.2f"
      % np.mean((prediction - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % LinearRegr.score(X, y))

# no error for the next part
LinearRegression_MAE = np.mean(np.absolute(prediction - y_test))
LinearRegression_MSE = np.mean((prediction -y_test)**2)
LinearRegression_R2 = r2_score(y_test, prediction)
print("Mean absolute error: %.2f" % LinearRegression_MAE)
print("Residual sum of squares (MSE): %.2f" % LinearRegression_MSE)
print("R2-score: %.2f" % LinearRegression_R2 )

# Using KNN
kNN = 4
neigh = KNeighborsClassifier(n_neighbors = kNN).fit(X_train,y_train)
neigh

# Prediction and Evaluation
prediction = neigh.predict(X_test)
prediction[0:5]

KNN_Accuracy_Score = metrics.accuracy_score(y_test, prediction)
KNN_JaccardIndex = metrics.jaccard_score(y_test, prediction)
KNN_F1_Score = metrics.f1_score(y_test, prediction)
KNN_Log_Loss = metrics.log_loss(y_test, prediction)

print("KNN Accuracy Score: {0:.3f}".format(KNN_Accuracy_Score))
print("KNN_JaccardIndex: {0:.3f}".format(KNN_JaccardIndex))
print("KNN F1 score: {0:.3f}".format(KNN_F1_Score))
print("KNN Log Loss: {0:.3f}".format(KNN_Log_Loss))

# Using Decision Tree

tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
tree.fit(X_train, y_train)

Tree_Accuracy_Score = metrics.accuracy_score(y_test, prediction)
Tree_JaccardIndex = metrics.jaccard_score(y_test, prediction)
Tree_F1_Score = metrics.f1_score(y_test, prediction)
Tree_Log_Loss = metrics.log_loss(y_test, prediction)

print("Tree accur_acy score: {0:.3f}".format(Tree_Accuracy_Score))
print("Tree JaccardIndex : {0:.3f}".format(Tree_JaccardIndex))
print("Tree_F1_Score : {0:.3f}".format(Tree_F1_Score))
print("Tree Log Loss : {0:.3f}".format(Tree_Log_Loss))

# Using Logistic Regression
LR = LogisticRegression(C=0.01, solver='liblinear')
LR.fit(X_train,y_train)

# Predictions and Evaluation
prediction = LR.predict(X_test)
LR_Accuracy_Score = metrics.accuracy_score(y_test, prediction)
LR_JaccardIndex = metrics.jaccard_score(y_test, prediction)
LR_F1_Score = metrics.f1_score(y_test, prediction)
LR_Log_Loss = metrics.log_loss(y_test, prediction)

print("LR accuracy score: {0:.3f}".format(LR_Accuracy_Score))
print("LR JaccardIndex: {0:.3f}".format(LR_JaccardIndex))
print("LR F1 Score: {0:.3f}".format(LR_F1_Score))
print("LR Log Loss: {0:.3f}".format(LR_Log_Loss))

# Using SVM
SVM = svm.SVC(kernel='linear')
SVM.fit(X_train, y_train)

# Predictions and Evaluation
prediction = SVM.predict(X_test)
SVM_Accuracy_Score = metrics.accuracy_score(y_test, prediction)
SVM_JaccardIndex = metrics.jaccard_score(y_test, prediction)
SVM_F1_Score = metrics.f1_score(y_test, prediction)
SVM_Log_Loss = metrics.log_loss(y_test, prediction)

print("SVM accuracy score : {0:.3f}".format(SVM_Accuracy_Score))
print("SVM jaccardIndex : {0:.3f}".format(SVM_JaccardIndex))
print("SVM F1_score : {0:.3f}".format(SVM_F1_Score))
print("SVM Log Loss : {0:.3f}".format(SVM_Log_Loss))

# table in a DataFrame, comparing the evaluations of the above models used
data = {'KNN':[KNN_Accuracy_Score, KNN_JaccardIndex, KNN_F1_Score, KNN_Log_Loss],
     'Tree':[Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score, Tree_Log_Loss],
     'LR':[LR_Accuracy_Score, LR_JaccardIndex, LR_F1_Score, LR_Log_Loss],
     'SVM':[SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score, SVM_Log_Loss]}
Report = pd.DataFrame(data=data, index=['Accuracy','Jaccard Index','F1-Score', 'LogLoss'])
Report
