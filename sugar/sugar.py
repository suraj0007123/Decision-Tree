import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\decision treeeee\Datasets_DTRF\Diabetes.csv")

data.isnull().sum()
data.dropna()
data.columns

data.rename(columns={' Number of times pregnant':'Pregnancies',' Plasma glucose concentration':'Glucose',' Diastolic blood pressure':'BloodPressure',' Triceps skin fold thickness':'SkinThickness',' 2-Hour serum insulin':'Insulin',' Body mass index':'BMI',' Diabetes pedigree function':'DiabetesPedigreefunction',' Age (years)':'Age',' Class variable':'Outcome'},inplace=True)

data.columns

data.dtypes

#data["default"]=lb.fit_transform(data["default"])
data.nunique()
data['Outcome'].unique()
data['Outcome'].value_counts()
colnames = list(data.columns)

predictors = colnames[0:8]
target = colnames[8]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy',max_depth=4)
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

data.columns


import matplotlib.pyplot as plt
from sklearn import tree

fig = plt.figure(figsize=(40,20))
fig = tree.plot_tree(model,
 feature_names= ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
 'BMI', 'DiabetesPedigreefunction', 'Age'], class_names= ['Non-Diabetic', 'Diabetic'], filled=True)
plt.title('Decision tree using Entropy on Diabetes',fontsize=22)
plt.savefig('DT_Entropy.pdf')

################### Random Forest ####################

data.info()

data.head()
predictors = data.loc[:, data.columns!="Outcome"]
type(predictors)

target = data["Outcome"]
type(target)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(predictors, target , test_size = 0.2 , random_state=7)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators = 500 , n_jobs=-1 , random_state = 42)
rf_clf.fit(x_train , y_train)
data.dtypes

from sklearn.metrics import accuracy_score , confusion_matrix
confusion_matrix(y_test , rf_clf.predict(x_test))
accuracy_score(y_test , rf_clf.predict(x_test))

## Evaluation on training data
confusion_matrix(y_train , rf_clf.predict(x_train))
accuracy_score(y_train , rf_clf.predict(x_train))

##Grid search cv
from sklearn.model_selection import GridSearchCV
rf_clf_grid = RandomForestClassifier(n_estimators= 500 , n_jobs = -1 , random_state = 42)
param_grid = {"max_features" : [4,5,6,7,8,9,10] , "min_samples_split":[2,3,10], "max_depth":[2,3,4]}
grid_search = GridSearchCV(rf_clf_grid , param_grid, n_jobs = -1, cv = 5 , scoring = "accuracy")

grid_search.fit(x_train , y_train)
grid_search.best_params_
cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score , confusion_matrix
confusion_matrix(y_test , cv_rf_clf_grid.predict(x_test))
accuracy_score(y_test, cv_rf_clf_grid.predict(x_test))

###Evaluation on training data
confusion_matrix(y_train , cv_rf_clf_grid.predict(x_train))
accuracy_score(y_train, cv_rf_clf_grid.predict(x_train))
