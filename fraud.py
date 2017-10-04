#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 23:26:38 2017

@author: julien
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from imblearn.over_sampling import ADASYN, SMOTE
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from keras import optimizers
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

t0 = dt.datetime.now()

dataset = pd.read_csv('creditcard.csv')

X = dataset.iloc[:,0:30].values
sc = StandardScaler()
X = sc.fit_transform(X)

Y = dataset.iloc[:,30].values

sns.countplot(Y)
sns.plt.title("Répartition des observations")

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

# plusieurs façon d'oversampler la classe minoritaire
smote = SMOTE()
adasyn = ADASYN()
X_adasyn, Y_adasyn = adasyn.fit_sample(X_train, Y_train)
X_SMOTE, Y_SMOTE = smote.fit_sample(X_train, Y_train)

XY_SMOTE = np.column_stack([X_SMOTE,Y_SMOTE])
XY_adasyn = np.column_stack([X_adasyn,Y_adasyn])

#shuffling
np.take(XY_SMOTE,np.random.permutation(XY_SMOTE.shape[0]),axis=0,out=XY_SMOTE)
np.take(XY_adasyn,np.random.permutation(XY_adasyn.shape[0]),axis=0,out=XY_adasyn)

X_SMOTE, Y_SMOTE = XY_SMOTE[:,:30], XY_SMOTE[:,30]
X_adasyn, Y_adasyn = XY_adasyn[:,:30], XY_adasyn[:,30]

def create_model(learn_rate=0.02, momentum=0.8, nesterov=True, activ='relu', output=30, nb_hidden=1, dropout=0.05, init='uniform'):
    # Création du réseau de neurones
    classifier = Sequential()
    classifier.add(Dense(input_dim = 30, output_dim = output, activation = activ))
    classifier.add(Dropout(dropout))
    for i in range(nb_hidden-1):
        classifier.add(Dense(output_dim = output, activation = activ))
        classifier.add(Dropout(dropout))
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
        
    optimizer = optimizers.SGD(lr=learn_rate, momentum=momentum, nesterov=nesterov)
    # Compilation
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#petits jeux de données pour entrainer le modèle plus rapidement et trouver les bons paramètres
#X_SMOTE = X_SMOTE[:20000,:]
#Y_SMOTE = Y_SMOTE[:20000]
#X_adasyn = X_adasyn[:20000,:]
#Y_adasyn = Y_adasyn[:20000]
    


def optim_param():    
    ##nnet   
    classifier = KerasClassifier(build_fn=create_model)
    
    #paramètres modèle
    learn_rate = [0.001, 0.01]
    momentum = [0.0, 0.2, 0.4, 0.8]
    momentum = [0.8,1.0,1.5,2.0]
    learn_rate = [0.01,0.02,0.05]
    nesterov = [True, False]
    
    output=[10, 20]
    activ = ['relu','tanh']
    init = ['normal', 'uniform']
    nb_hidden = [1,2]
    dropout = [0.0,0.05,0.1,0.15,0.2]
    
    #paramètres fit
    epochs = [10, 20, 50]
    batches = [5, 10, 20]
    
    #tests sur l'optimizer
    param_grid = dict(learn_rate=learn_rate, momentum=momentum, nesterov=nesterov, epochs=[10], batch_size=batches)
#     Résultats:
#        SMOTE : Best: 0.994400 using {'batch_size': 10, 'epochs': 10, 'learn_rate': 0.01, 'momentum': 0.8, 'nesterov': True}
#        ADASYN : Best: 0.962550 using {'batch_size': 10, 'epochs': 10, 'learn_rate': 0.01, 'momentum': 0.8, 'nesterov': True}
#        on relance le programme avec d'autres momentum et lr !
#         {'batch_size': 10,
#         'epochs': 10,
#         'learn_rate': 0.02,
#         'momentum': 0.8,
#         'nesterov': True}
#    
    
    #tests sur la structure du réseau de neurones
    param_grid = dict(learn_rate=[0.02], momentum=[0.8], nesterov=[True], epochs=[10], batch_size=[10], output=output, activ=activ, init=init, nb_hidden=nb_hidden, dropout=dropout)
#    SMOTE : Best: 0.995250 using {'activ': 'relu', 'batch_size': 10, 'dropout': 0.05,
#     'epochs': 10, 'init': 'uniform', 'learn_rate': 0.02, 'momentum': 0.8, 'nb_hidden': 1, 'nesterov': True, 'output': 20}
    
    #encore un test
    param_grid = dict(epochs=[10,20], batch_size=[10,20], output=[20,30], nb_hidden=[1,2])
    #SMOTE : Best: 0.996350 using {'batch_size': 10, 'epochs': 20, 'nb_hidden': 2, 'output': 30}
    
    #un autre test...
    param_grid = dict(epochs=[20], batch_size=[10], output=[20,30,50], nb_hidden=[1,2,3])
    #SMOTE : Best: 0.997200 using {'batch_size': 10, 'epochs': 20, 'nb_hidden': 1, 'output': 30}
    
    #test final !
    param_grid = dict(epochs=[10,20,50,100], batch_size=[10])
    epochs = 100
    
    
    grid = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
    
    #full paramètres : trop long à faire tourner !!!
    param_grid = dict(epochs=epochs, batch_size=batches, init=init, learn_rate=learn_rate, momentum=momentum, nesterov=nesterov, activ=activ, output=output, nb_hidden = nb_hidden, dropout=dropout)
    grid = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
    
    #SMOTE
    grid_result_smote = grid.fit(X_SMOTE, Y_SMOTE)
    # summarize results
    results_SMOTE = ("SMOTE : Best: %f using %s" % (grid_result_smote.best_score_, grid_result_smote.best_params_))
    means = grid_result_smote.cv_results_['mean_test_score']
    stds = grid_result_smote.cv_results_['std_test_score']
    params = grid_result_smote.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        results_SMOTE += "\n%f (%f) with: %r" % (mean, stdev, param)
        
    #ADASYN
    grid_result_adasyn = grid.fit(X_adasyn, Y_adasyn)
    # summarize results
    results_ADASYN = "ADASYN : Best: %f using %s" % (grid_result_adasyn.best_score_, grid_result_adasyn.best_params_)
    means = grid_result_adasyn.cv_results_['mean_test_score']
    stds = grid_result_adasyn.cv_results_['std_test_score']
    params = grid_result_adasyn.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        results_ADASYN += "\n%f (%f) with: %r" % (mean, stdev, param)
    
    
    classifier.fit(X_SMOTE, Y_SMOTE, epochs=100, batch_size=10)
    #results with 0.5 threshold
    Y_pred_smote = np.round(classifier.predict(X_test))
    cm_smote = confusion_matrix(Y_test, Y_pred_smote)
    print("SMOTE :", cm_smote)
    Y_pred_smote = classifier.predict(X_test) # pour faire la courbe ROC
    
    classifier.fit(X_adasyn, Y_adasyn, epochs=100, batch_size=10)
    Y_pred_adasyn = np.round(classifier.predict(X_test))
    cm_adasyn = confusion_matrix(Y_test, Y_pred_adasyn)
    print("ADASYN :", cm_adasyn)
    Y_pred_adasyn = classifier.predict(X_test) # pour faire la courbe ROC
    
    t1 = dt.datetime.now()
    
    diff = t1 - t0
    print(diff)
    
    
    #random forest
    
    random_forest = RandomForestClassifier()
    
    rf_param_grid = dict(n_estimators=[10,100,1000], max_features=[2, 4, 6, 8], bootstrap=[False, True])
    rf_grid_search = GridSearchCV(random_forest, rf_param_grid, cv=5,scoring='neg_mean_squared_error')
    rf_grid_search.fit(X_SMOTE, Y_SMOTE)
    
    rf_grid_search.best_params_
    rf_grid_search.best_estimator_
    
    cvres = rf_grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    	print(np.sqrt(-mean_score), params)
        
    t1 = dt.datetime.now()
    diff = t1 - t0
    print(diff)
    
    {'bootstrap': False, 'max_features': 8, 'n_estimators': 1000}
    
    
    #XGBoost
    
    xgb_classifier = xgb.XGBClassifier()
    xgb_param_grid = dict(n_estimators=[10,100,1000], reg_alpha = [0,0.1,0.2,0.5], reg_lambda = [0.5,1,1.5], learning_rate = [0.01,0.1,0.2])
    xgb_grid_search = GridSearchCV(xgb_classifier, xgb_param_grid, cv=5,scoring='neg_mean_squared_error')
    xgb_grid_search.fit(X_SMOTE, Y_SMOTE)
    xgb_grid_search.best_estimator_
    xgb_grid_search.best_params_
    
    cvres = xgb_grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

#optim_param()

#les prédictions pour chaque estimateur

##### neural network
nnet_classifier = Sequential()
nnet_classifier.add(Dense(input_dim = 30, output_dim = 30, activation = 'relu'))
nnet_classifier.add(Dropout(0.05))
nnet_classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    
optimizer = optimizers.SGD(lr=0.02, momentum=0.8, nesterov=True)
# Compilation
nnet_classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

nnet_classifier.fit(X_SMOTE, Y_SMOTE, batch_size=10, epochs=100)
nnet_SMOTE = nnet_classifier.predict(X_test)
nnet_SMOTE_probas = nnet_classifier.predict_proba(X_test)

nnet_classifier.fit(X_adasyn, Y_adasyn, batch_size=10, epochs=100)
nnet_ADASYN = nnet_classifier.predict(X_test)


##### random forest
rf_classifier = RandomForestClassifier(n_estimators=1000, max_features=8, bootstrap=False)
rf_classifier.fit(X_SMOTE, Y_SMOTE)
rf_SMOTE = rf_classifier.predict(X_test)

rf_classifier.fit(X_adasyn, Y_adasyn)
rf_ADASYN = rf_classifier.predict(X_test)

diff = dt.datetime.now() - t0

##### xgboost
xgb_classifier = xgb.XGBClassifier(learning_rate=0.2, n_estimators=1000, reg_alpha=0, reg_lambda=1, nthread=-1)


xgb_classifier.fit(X_SMOTE, Y_SMOTE)
xgb_SMOTE = xgb_classifier.predict(X_test)

xgb_classifier.fit(X_adasyn, Y_adasyn)
xgb_ADASYN = xgb_classifier.predict(X_test)


# comparaison des modèles
plt.title('Receiver Operating Characteristic')

fpr, tpr, _ = roc_curve(Y_test, rf_SMOTE)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, c='red', label = 'AUC = %0.2f' % roc_auc)

fpr, tpr, _ = roc_curve(Y_test, rf_ADASYN)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, c='blue', label = 'AUC = %0.2f' % roc_auc)

fpr, tpr, _ = roc_curve(Y_test, xgb_SMOTE)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, c='black', label = 'AUC = %0.2f' % roc_auc)

fpr, tpr, _ = roc_curve(Y_test, xgb_ADASYN)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, c='brown', label = 'AUC = %0.2f' % roc_auc)

fpr, tpr, _ = roc_curve(Y_test, nnet_SMOTE)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, c='green', label = 'AUC = %0.2f' % roc_auc)

fpr, tpr, _ = roc_curve(Y_test, nnet_ADASYN)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, c='pink', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#### meilleurs résultats : SMOTE + nnet
for i in range(21):
    threshold = i/20.0
    Y_pred = np.zeros(len(Y_test))
    for i in range(len(Y_pred)):
        if nnet_SMOTE_probas[i] >= threshold:
            Y_pred[i] = 1
    cm = confusion_matrix(Y_test, Y_pred)
    print("True positives :", cm[1,1], "; False negatives :", cm[1,0], "; False positives :", cm[0,1])
    
sns.heatmap(cm,annot=True)