import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\decision treeeee\Datasets_DTRF\HR_DT.csv")

data.isnull().sum()
data.dropna()
data.columns

data = pd.get_dummies(data,columns = ["Position of the employee"], drop_first = True)

#### Drop the column
data.columns

lb = LabelEncoder()
data[" monthly income of employee"] = lb.fit_transform(data[" monthly income of employee"])
data['Position of the employee_C-level'] = lb.fit_transform(data['Position of the employee_C-level'])
data['Position of the employee_CEO'] = lb.fit_transform(data['Position of the employee_CEO'])
data['Position of the employee_Country Manager'] = lb.fit_transform(data['Position of the employee_Country Manager'])
data['Position of the employee_Junior Consultant'] = lb.fit_transform(data['Position of the employee_Junior Consultant'])
data['Position of the employee_Manager'] = lb.fit_transform(data['Position of the employee_Manager'])
data['Position of the employee_Partner'] = lb.fit_transform(data['Position of the employee_Partner'])
data['Position of the employee_Region Manager'] = lb.fit_transform(data['Position of the employee_Region Manager'])
data['Position of the employee_Senior Consultant'] = lb.fit_transform(data['Position of the employee_Senior Consultant'])
data['Position of the employee_Senior Partner'] = lb.fit_transform(data['Position of the employee_Senior Partner'])

#data["default"]=lb.fit_transform(data["default"])
data.nunique()
data[' monthly income of employee'].unique()
data[' monthly income of employee'].value_counts()
colnames = list(data.columns)

predictors = colnames[0:2]
target = colnames[2]


# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
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
 feature_names= ['no of Years of Experience of employee', ' monthly income of employee',
        'Position of the employee_C-level', 'Position of the employee_CEO',
        'Position of the employee_Country Manager',
        'Position of the employee_Junior Consultant',
        'Position of the employee_Manager', 'Position of the employee_Partner',
        'Position of the employee_Region Manager',
        'Position of the employee_Senior Consultant',
        'Position of the employee_Senior Partner'], class_names= ['Risky', 'Good'], filled=True)
plt.title('Decision tree using Entropy HR Data',fontsize=22)
plt.savefig('DT_Entropy.pdf')



################### Random Forest ####################

data.info()

data.head()

predictors = data.loc[:, data.columns!=" monthly income of employee"]
type(predictors)

target = data[" monthly income of employee"]
type(target)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(predictors, target , test_size = 0.3 , random_state=0)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators = 10 , n_jobs= 1 , random_state = 42)
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
rf_clf_grid = RandomForestClassifier(n_estimators= 350 , n_jobs = -1 , random_state = 42)
param_grid = {"max_features" : [2,3,4,5,6,7,8] , "min_samples_split":[2,3,10], "max_depth":[2,3,4] }
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
