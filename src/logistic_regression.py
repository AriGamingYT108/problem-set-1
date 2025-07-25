'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


# Your code here

df_arrests = pd.read_csv("data/df_arrests.csv", parse_dates=["arrest_date"])

df_arrests_train, df_arrests_test = train_test_split(
    df_arrests,
    test_size=0.3,
    shuffle=True,
    stratify=df_arrests['y'],
    random_state=42
)

print("Train set shape:", df_arrests_train.shape)
print("Test set shape: ", df_arrests_test.shape)

features = ['num_fel_arrests_last_year', 'current_charge_felony']
print("Using features:", features)

X_train = df_arrests_train[features]
y_train = df_arrests_train['y']

X_test  = df_arrests_test[features]
y_test  = df_arrests_test['y']

print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)

param_grid = {'C': [0.01, 0.1, 1.0]}
lr_model = lr(solver='liblinear')
gs_cv = GridSearchCV(lr_model, param_grid, cv=5)
gs_cv.fit(X_train, y_train)

# Inspect best C
best_C = gs_cv.best_params_['C']

if best_C == min(param_grid['C']):
    reg_level = "most regularized"
elif best_C == max(param_grid['C']):
    reg_level = "least regularized"
else:
    reg_level = "middle regularization"

print(f"What was the optimal value for C? {best_C}")
print(f"This corresponds to {reg_level}.\n")


df_arrests_test['pred_lr'] = gs_cv.predict_proba(X_test)[:, 1]

# Quick check
print("Test set with predictions:")
print(df_arrests_test[['y', 'pred_lr']].head())

df_arrests_train.to_csv("data/df_arrests_train.csv", index=False)
df_arrests_test.to_csv("data/df_arrests_test.csv",  index=False)

df_train = pd.read_csv("data/df_arrests_train.csv")
df_test  = pd.read_csv("data/df_arrests_test.csv")

# Features for the tree
X_train = df_train[['num_fel_arrests_last_year','current_charge_felony']]
y_train = df_train['y']
X_test  = df_test[['num_fel_arrests_last_year','current_charge_felony']]
y_test  = df_test['y']