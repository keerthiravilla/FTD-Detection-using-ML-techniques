# -*- coding: utf-8 -*-
#Import


import os
import sys
import glob
import nibabel
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# from google.colab import drive
# drive.mount('/content/drive')

#Variables

image_size = np.array([61,73,61]) 

#Constants

#--OLD
# PAZIENTI_FOLDER='../imgs/ALFF/1/'
# CONTROLLI_FOLDER='../imgs/ALFF/3/'
PAZIENTI_FOLDER=sys.argv[1]
CONTROLLI_FOLDER=sys.argv[2]


#Load Dataset

#Patients


# os.chdir(PAZIENTI_FOLDER) 
images_pazienti = glob.glob(PAZIENTI_FOLDER+'*.nii', recursive=True)
num_pazienti = len(images_pazienti) 

X_pazienti = np.zeros((num_pazienti,np.product(image_size)))
t = 0
for fMRI in images_pazienti:
  file_nii = nibabel.load(fMRI) 
  img = np.array(file_nii.dataobj)
  X_pazienti[t,:] = np.reshape(img,(1,np.product(image_size)))
  t = t + 1

y_pazienti = np.ones((num_pazienti,1))

#Controls

from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer


# os.chdir(CONTROLLI_FOLDER) 
images_controlli = glob.glob(CONTROLLI_FOLDER+'*.nii', recursive=True) 
num_controlli = len(images_controlli)

X_controlli = np.zeros((num_controlli,np.product(image_size)))
t = 0
for fMRI in images_controlli:
  file_nii = nibabel.load(fMRI)
  img = np.array(file_nii.dataobj)
  X_controlli[t,:] = np.reshape(img,(1,np.product(image_size)))
  t = t + 1

y_controlli = np.zeros((num_controlli,1))

#Union

X = np.concatenate((X_pazienti,X_controlli))
y = np.concatenate((y_pazienti,y_controlli))

#Preprocessing

initial_num_cols = X.shape[1]

mask = (X == 0).all(0)
column_indices = np.where(mask)[0]
X = X[:,~mask]

final_num_cols = X.shape[1]

print(str(initial_num_cols - final_num_cols) + ' columns were dropped from the dataset')

#Pipeline - search for best preprocessing

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer, StandardScaler, MinMaxScaler, MaxAbsScaler

print(PAZIENTI_FOLDER)
print(CONTROLLI_FOLDER)

# PARAMETERS OF RandomForestClassifier
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# Pipeline - Param grid
param_grid = {
    #"pca__n_components": [20, 25, 30, 34],
    
    # Create the random grid
    'clf__n_estimators': n_estimators,
    'clf__max_features': max_features,
    'clf__max_depth': max_depth,
    'clf__min_samples_split': min_samples_split,
    'clf__min_samples_leaf': min_samples_leaf,
    'clf__bootstrap': bootstrap
}


scalers = [StandardScaler(), RobustScaler(), QuantileTransformer(), MinMaxScaler(), MaxAbsScaler()]
for scaler in scalers:
  print("Scaler: ", scaler)
  pipe = Pipeline(steps=[
                       ('scaler', scaler), 
                       #('pca', PCA()), 
                       ('clf', RandomForestClassifier())
                      ])


  #RandomizedSearchCV
  search = RandomizedSearchCV(pipe, param_grid, n_jobs=-1, random_state=1, scoring='accuracy')
  search.fit(X, y.ravel())

  y_pred = search.predict(X)

  print("f1 (RndSrc): %.3f" %f1_score(y.ravel(), y_pred))
  print("Accuracy (RndSrc): %.3f" %accuracy_score(y.ravel(), y_pred))

  print("\t--> Best score (CV score=%0.3f):" % search.best_score_)
  print("\t--> Best params: ", search.best_params_)

  

  #SSS
  # Evaluate model
  sss = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=1)
  scores = cross_val_score(pipe, X, y.ravel(), scoring='accuracy', cv=sss, n_jobs=-1)

  # Print
  print('\nAccuracy (SSS): %.2f (%.2f)' % (np.mean(scores), np.std(scores)))
  print("\t--> ", search.cv_results_['params'][search.best_index_])

  
  
  #Leave-One-Out
  # Evaluate model
  scores = cross_val_score(pipe, X, y.ravel(), scoring='accuracy', cv=LeaveOneOut(), n_jobs=-1)

  # Print
  print('\nAccuracy (Leave-One-Out): %.2f (%.2f)' % (np.mean(scores), np.std(scores)))

  print('\n############################\n')
