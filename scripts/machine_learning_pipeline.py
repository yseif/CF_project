import pickle
import pandas as pd
from collections import defaultdict, OrderedDict, Counter
import time
from glob import glob
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import re
import os
import scipy

from sklearn.model_selection import StratifiedShuffleSplit
import random

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_validate, cross_val_predict,  GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import sys

from tqdm import tqdm

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix
import matplotlib.patches
import matplotlib.font_manager
import matplotlib

def get_holdout_dataset(FT_LRI, input_matrix, directory, keyword = ''):

    labels = np.array(FT_LRI['cystic_fibrosis_status'])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    sss.get_n_splits(input_matrix, labels)

    for train_index, holdout_index in sss.split(input_matrix, labels):
        data_FS, data_holdout = input_matrix[train_index,:], input_matrix[holdout_index,:]
        labels_FS, labels_holdout = labels[train_index], labels[holdout_index]

    rand = np.array([random.randint(0, 1) for _ in range(data_FS.shape[0])])
    data_FS = np.vstack((data_FS.T, rand)).T

    output_d = '%s/machine_learning/input_data_%s.npz'%(directory, keyword) if keyword != '' else '%s/machine_learning/input_data.npz'%(directory)
    np.savez(output_d,
            train_index = train_index, 
            holdout_index = holdout_index,
             rand = rand
            )
    return (data_FS, data_holdout, labels_FS, labels_holdout)


def run_feature_selection_part1(data_FS, labels_FS, directory = '', keyword = ''):
    
    models = {'LinearSVC':LinearSVC(penalty="l1", loss="squared_hinge", dual=False, max_iter = 10000, class_weight="balanced"), 
              'LogisticRegression':LogisticRegression(penalty= "l1", dual = False),
              'RandomForestClassifier':RandomForestClassifier(n_estimators = 100)
             }
    all_rows = defaultdict(list)

    for model_id, clf_i in models.items():
        for c in [0.5, 1, 2]:
            if model_id == 'RandomForestClassifier':
                clf = clf_i.fit(data_FS, labels_FS)
                coef_d = dict(zip(range(data_FS.shape[1]), clf.feature_importances_))
                d = {x:y for x,y in coef_d.items() if abs(y) > abs(coef_d[data_FS.shape[1] -1])}
            else:
                clf_i.C = c
                clf = clf_i.fit(data_FS, labels_FS)
                coef_d = dict(zip(range(data_FS.shape[1]), clf.coef_.tolist()[0]))
                d = {x:y for x,y in coef_d.items() if abs(y) > abs(coef_d[data_FS.shape[1] -1])}
            all_rows['%s - C=%d - random'%(model_id,c)] = d
    if directory != '':
        output_d = '%s/machine_learning/ML_trial_run_2_%s.npz'%(directory, keyword) if keyword != '' else '%s/machine_learning/ML_trial_run_2.npz'%(directory)
        pickle.dump(all_rows, open(output_d, 'wb'))
    
    return all_rows




def train_ML_models_part1(data_FS, labels_FS, all_rows_FS, directory = '', keyword = ''):
    
    training_models = {
        'LogisticRegression':LogisticRegression(solver = 'lbfgs', max_iter = 10000),
        'ExtraTreesClassifier':ExtraTreesClassifier(n_estimators = 100),
        'RandomForest':RandomForestClassifier(n_estimators = 100),
        'LinearSVC':LinearSVC(penalty="l1", dual=False, max_iter = 10000),
        'LinearSVC(L-2)':LinearSVC(penalty="l2", max_iter = 10000)
    }

    rows = []
    for training_model_id, clf in training_models.items():
        for selection_model_id, features in all_rows_FS.items():
            selected_features = list(features.keys())
            skf = StratifiedKFold(n_splits=10)
            scores = cross_validate(clf, data_FS[:,selected_features], labels_FS, cv=skf, return_train_score = True)
            predicted = cross_val_predict(clf, data_FS[:,selected_features], labels_FS, cv=skf)
            overfitting_proxy = (np.median(scores['train_score']) - np.median(scores['test_score']))*100

            rows.append({'ML model':training_model_id, 'FS model':selection_model_id, 
                         'F-1':metrics.f1_score(labels_FS, predicted), 
                         'Accuracy':metrics.accuracy_score(labels_FS, predicted),
                        'ROC-AUC score': metrics.roc_auc_score(labels_FS, predicted), 
                        'jaccard':metrics.jaccard_similarity_score(labels_FS, predicted),
                        'MI':metrics.normalized_mutual_info_score(labels_FS, predicted),
                         'brier_score_loss':metrics.brier_score_loss(labels_FS, predicted),
                         'precision':metrics.precision_score(labels_FS, predicted),
                         'recall':metrics.recall_score(labels_FS, predicted),
                         'No. selected features':len(selected_features),
                         'Overfitting proxy':overfitting_proxy
                        })

    scores_res = pd.DataFrame(rows)
    scores_res = scores_res.sort_values(by = 'Overfitting proxy')
    if directory != '':
        output_d = '%s/machine_learning/ML_trial_run_2_scores_res_%s.npz'%(directory, keyword) if keyword != '' else '%s/machine_learning/ML_trial_run_2_scores_res.npz'%(directory)
        scores_res.to_csv(output_d)
    return scores_res

def run_feature_selection_part2(data_FS, labels_FS, bootstrap_no = 1000, directory = '', keyword = '', C = 1):

    clf_i = LinearSVC(penalty="l1", C = C, loss="squared_hinge", dual=False, max_iter = 10000, class_weight="balanced")

    stratified_shuffle = StratifiedShuffleSplit(n_splits = bootstrap_no, test_size=0.1, random_state=0)
    stratified_shuffle.get_n_splits(data_FS, labels_FS)
    bootstrap_rows = {}

    i = 0
    with tqdm(total=int(bootstrap_no/10), leave = False, file=sys.stdout, initial = 0) as pbar:
        for train_index, test_index in stratified_shuffle.split(data_FS, labels_FS):
            clf = clf_i.fit(data_FS[train_index,:], labels_FS[train_index])
            coef_d = dict(zip(range(data_FS.shape[1]), clf.coef_.tolist()[0]))
            d = {x:y for x,y in coef_d.items() if abs(y) > abs(coef_d[data_FS.shape[1] -1])}
            bootstrap_rows['Try_%d'%i] = d 
            i += 1
            if i%10 == 0:
                pbar.update(1)

    if directory != '':
        output_d = '%s/machine_learning/bootstrap_rows_SVC_trial_2_%s.npz'%(directory, keyword) if keyword != '' else '%s/machine_learning/bootstrap_rows_SVC_trial_2.npz'%(directory)
        
        pickle.dump(bootstrap_rows, open(output_d, 'wb'))
        
    return bootstrap_rows


def assess_performance(data_FS,labels_FS, bootstrap_rows, no_features = 200, mi_cutoff = 0.91, directory = '', keyword = ''):
    bootstrap_res = pd.DataFrame(bootstrap_rows).fillna(0)
    sorted_features = abs(bootstrap_res).sum(axis = 1).sort_values(ascending = False)
    kappa = defaultdict(dict)

    skf = StratifiedKFold(n_splits=10)
    clf = LinearSVC(penalty="l2", max_iter = 10000, class_weight = 'balanced')
    scores_FS = []

    for i in range(10,no_features,5):
        selected_features = sorted_features[:i].index
        scores = cross_validate(clf, data_FS[:, selected_features], labels_FS, cv=skf, return_train_score = True)
        predicted = cross_val_predict(clf, data_FS[:,selected_features], labels_FS, cv=skf)
        overfitting_proxy = (np.median(scores['train_score']) - np.median(scores['test_score']))


        scores_FS.append({
                     'F-1':metrics.f1_score(labels_FS, predicted), 
                     'Accuracy':metrics.accuracy_score(labels_FS, predicted),
    #                 'Accuracy (test)':metrics.accuracy_score(labels_holdout, predicted_test),
                    'ROC-AUC score': metrics.roc_auc_score(labels_FS, predicted), 
                    'jaccard':metrics.jaccard_similarity_score(labels_FS, predicted),
                    'MI':metrics.normalized_mutual_info_score(labels_FS, predicted, average_method='arithmetic'),
                     'brier_score_loss':metrics.brier_score_loss(labels_FS, predicted),
                     'precision':metrics.precision_score(labels_FS, predicted),
                     'recall':metrics.recall_score(labels_FS, predicted),
                     'No. selected features':len(selected_features),
                     'Overfitting proxy':overfitting_proxy
                    })
        
    scores_FS_df = pd.DataFrame(scores_FS)
    
    if directory != '':
        output_d = '%s/machine_learning/bootstrap_scores_SCV_trial_2_%s.npz'%(directory, keyword) if keyword != '' else '%s/machine_learning/bootstrap_scores_SCV_trial_2.npz'%(directory)
        scores_FS_df.to_csv(output_d)
    
    for score_type in ['ROC-AUC score', 'Accuracy','F-1', 'MI', 'precision', 'recall', 'brier_score_loss', 'jaccard', 'Overfitting proxy']:
        plt.plot(scores_FS_df['No. selected features'], scores_FS_df[score_type], '.', label = score_type)
        
    plt.legend(bbox_to_anchor = (1,1))

    t = min(scores_FS_df.loc[scores_FS_df['MI'] > mi_cutoff]['No. selected features'])

    plt.axvline(t, color = 'grey')
    plt.xlabel('No. features')
    plt.ylabel('Metric')
    plt.title('Sensitivity analysis of feature set for ML training')

    return scores_FS_df


def generate_figures(data_FS, labels_FS, data_holdout, labels_holdout, selected_features_final, species):
    selected_features = selected_features_final.index
    clf = LinearSVC(penalty="l2", max_iter = 10000, class_weight = 'balanced')

    matplotlib.rcParams.update({'font.size': 15, 'font.weight':'normal'})
    fig, axes = plt.subplots(1,2, figsize = (18, 6))

    lw = 2
    y_score = clf.fit(data_FS[:,selected_features], labels_FS).decision_function(data_holdout[:,selected_features])
    fpr, tpr, _ = roc_curve(labels_holdout, y_score)
    roc_auc_multi = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label='ROC curve (area = %f)'%(round(roc_auc_multi, 3)))
    axes[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver operating characteristic')
    axes[0].legend(loc="lower right")

    predicted = clf.fit(data_FS[:,selected_features], labels_FS).predict(data_holdout[:,selected_features])
    conf_m = confusion_matrix(labels_holdout,predicted)

    axes[1].axvline(0)
    axes[1].axhline(0)
    # ax=axes[1].gca()

    for count,indices in enumerate([[1,1], [1,0], [0,0], [0,1]]):
        r1 = np.log(conf_m[indices[0]][indices[1]])
        c1 = matplotlib.patches.Arc((0,0), r1, r1, angle = 90, theta1 = 90*(count), theta2 = 90*(count+1))
        axes[1].add_patch(c1)

    axes[1].text(-2.2, 2.2, '$\it{n}$= %d'%conf_m[1,1])
    axes[1].text(1.6, 2.2, '$\it{n}$= %d'%conf_m[0,1])
    axes[1].text(1.6, -2.2, '$\it{n}$= %d'%conf_m[0,0])
    axes[1].text(-2.2, -2.2, '$\it{n}$= %d'%conf_m[1,0])

    axes[1].axis('scaled')
    axes[1].set_xticks([-1, 1])
    axes[1].set_xticklabels(['Patients with \ncystic fibrosis','Patients without \n Cystic Fibrosis'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_yticks([-1, 1])
    axes[1].set_yticklabels(['Patients without \ncystic fibrosis','Patients with \n Cystic Fibrosis'])
    axes[1].set_ylabel('Observed')
    axes[1].set_title('Confusion matrix')

    fig.suptitle(species, y = 1.1, fontsize = 30)