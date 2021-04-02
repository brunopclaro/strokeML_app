import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Get the data

df = pd.read_csv('avc_dataML.csv',sep=";")

#Dataset split into independent x and dependent y variables
X = df.iloc[:, 0:25].values
Y = df.iloc[:,-1].values

#Dataset slipt into 75% training and 25% testing

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)

#Fine-tuning hyper-parameters

from sklearn.model_selection import GridSearchCV

RandomForestClassifier = RandomForestClassifier()

param_grid =[
    {'n_estimators': [3, 10, 30], 'max_features': [5,10,15,20]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [5,10,15]},
]

grid_search = GridSearchCV(RandomForestClassifier, param_grid,scoring="neg_mean_squared_error",return_train_score=True,cv=3)


#Create and train the model
grid_search.fit(X_train, Y_train)

#Feature importance

numerics=['float64',"int64"]

num_features= list(df.select_dtypes(include=numerics))

feature_importance = grid_search.best_estimator_.feature_importances_

FeatImp=pd.DataFrame(sorted(zip(feature_importance, num_features),reverse=True))

FeatImp.columns = ["Importance", "Features"]


#Show the model metrics

Model_metrics = accuracy_score(Y_test, grid_search.best_estimator_.predict(X_test))


