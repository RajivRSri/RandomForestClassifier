"""
Classification problem to predict churn for a telecom dataset.
"""

#Read dataset
import os
import pandas as pd
import numpy as np
import seaborn as sns

os.chdir ("C:/R")
data = pd.read_csv("telecom.csv")

#Check loaded data
data.head()
data.shape
data.info()

#Exploratory Data Analysis
data.isnull().values.any()
sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = "viridis")

#Check for imbalanced dataset
data.churn.value_counts()
sns.set_style("whitegrid")
sns.countplot(x = "churn", data = data, palette = "RdBu_r")

#Perform Lable Encoding
from sklearn.preprocessing import LabelEncoder
lblEncoder = LabelEncoder()
data.churn = lblEncoder.fit_transform(data.churn)
data["international plan"] = lblEncoder.fit_transform(data["international plan"])
data["voice mail plan"] = lblEncoder.fit_transform(data["voice mail plan"])

#Perform Mean Encoding on Categorical Variable
data.state.value_counts()

states = data.state.sort_values().unique()
states.shape

mean_churn = data.groupby(["state"])["churn"].mean()
mean_churn.shape

myDict = {}
for index in range( len(states)):
    myDict.update({states[index] : mean_churn[index]})

def meanChurn(val):
    return(myDict[val])

data["state"] = data.state.apply(lambda X : meanChurn(X))

#Drop Phone number variable
data = data.drop("phone number", axis = "columns")

#Divide dataset into dependent and independent sets
X = data.drop("churn", axis = "columns")
Y = data["churn"]
X.shape
#Handle imbalanced dataset
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state = 42)
X_res, Y_res = smk.fit_sample(X, Y)
print(X_res.shape, Y_res.shape)

#divide dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, train_size = 0.7, random_state = 0)

#Build Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 200, criterion = "entropy", n_jobs = -1, random_state = 20)
model.fit(X_train, Y_train)
model


#Hyperparameter Tuning using RandomizedSerchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

new_model = RandomForestClassifier()
params = {"max_depth" : [3, 5, 10, None],
          "n_estimators" : [100, 200, 300, 400, 500],
          "max_features" : randint(1, 20),
          "criterion" : ["entropy", "gini"],
          "bootstrap" : [True, False]}

randomizedSearchCV = RandomizedSearchCV(new_model,
                                    param_distributions = params,
                                    n_iter = 40,
                                    cv = 10,
                                    n_jobs = -1)
randomizedSearchCV.fit(X_train, Y_train)
randomizedSearchCV.best_params_

#Build RandomForest model again with best parameters
new_model = RandomForestClassifier(max_depth = None,
                                   n_estimators = 300,
                                   max_features = 2,
                                   criterion = "gini",
                                   bootstrap = False,
                                   n_jobs = -1,
                                   random_state = 40)

#Check model accuracy
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(new_model, X_res, Y_res, cv = 10)
accuracy.mean()

