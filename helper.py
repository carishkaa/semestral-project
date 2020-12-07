#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_regression
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV


def feature_selection_st_test(X, y, k):
    """
    selected_features = feature_selection_st_test(X, y, k)

    Feature selection based on the chi-squared (chi²) statistical test

    :param X:                   observations, (number_of_patients x number_of_genes) array
    :param y:                   labels (0 (OT) or 1 (CR)), (number_of_patients x 1) array
    :param k:                   number of features to select
    :return selected_features:  return k best features
    """
    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=k).fit(X, y)
    dfscores = pd.DataFrame(bestfeatures.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    # concat two dataframes for better visualization
    featureScores1 = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores1.columns = ['Genes', 'Score']  #naming the dataframe columns
    # print(featureScores1.nlargest(k,'Score'))
    selected_features = featureScores1.nlargest(k, 'Score')['Genes'].to_numpy()
    return selected_features


def feature_selection_extra_trees(X, y, k):
    """
    selected_features = feature_selection_extra_trees(X, y, k)

    Estimate the importance of features (using decision trees) and select best

    :param X:                   observations, (number_of_patients x number_of_genes) array
    :param y:                   labels (0 (OT) or 1 (CR)), (number_of_patients x 1) array
    :param k:                   number of features to select
    :return selected_features:  return k best features
    """
    model = ExtraTreesClassifier()
    model.fit(X, y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    selected_features = feat_importances.nlargest(k).index.to_numpy()
    return selected_features


def feature_selection_l1(X, y, k):
    """
    selected_features = feature_selection_l1(X, y, k)

    Selecting features using Lasso regularisation (L1)

    :param X:                   observations, (number_of_patients x number_of_genes) array
    :param y:                   labels (0 (OT) or 1 (CR)), (number_of_patients x 1) array
    :param k:                   number of features to select
    :return selected_features:  return k best features
    """
    scaler = StandardScaler()
    scaler.fit(X.fillna(0))
    model = LogisticRegression(C=1, penalty='l1', solver='liblinear')
    sel_ = SelectFromModel(model, max_features=k)
    sel_.fit(scaler.transform(X.fillna(0)), y)
    selected_features = X.columns[(sel_.get_support())]
    return selected_features


def feature_selection_random(X, k):
    """
    selected_features = feature_selection_random(X, k)

    Random feature selection

    :param X:                   observations, (number_of_patients x number_of_genes) array
    :param k:                   number of features to select
    :return selected_features:  return k random features
    """
    selected_features = X.sample(n=k, axis='columns').columns
    return selected_features


def group_data(cv_result):
    """
    data_groups = group_data(cv_result)

    Group data for boxplots by feature selection method

    :param cv_result:      cross-validation results
    :return groups:        return data grouped by feature selection methods
    """
    num_methods = 3
    groups = [[], [], []]
    for _, res in cv_result.items():
        tr = np.array(res).T.tolist()
        for ind in range(num_methods):
            groups[ind].append(tr[ind])
    return groups


def show_cv(data_groups, genes_range, methods):
    """
    show_cv(data_groups, genes_range, methods)

    Show cross-validation results using boxplot

    :param data_groups:     grouped by feature selection methods data
    :param genes_range:     list of numbers of selected genes
    :param methods:         list of strings of used feature selection methods
    """

    ticks = [str(k) for k in genes_range]

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)

    offset = [-0.5, 0.0, 0.5]
    colors = ['#D7191C', '#2C7BB6', '#2CA25F']

    for ind in range(len(data_groups)):
        bpl = plt.boxplot(data_groups[ind], positions=np.array(range(len(data_groups[ind]))) * 2.0 + offset[ind],
                          sym='', widths=0.4)
        set_box_color(bpl, colors[ind])

    # draw temporary red and blue lines and use them to create a legend
    for ind in range(len(methods)):
        plt.plot([], c=colors[ind], label=methods[ind])
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Number of genes')
    plt.ylabel('AUC')
    plt.tight_layout()
    plt.savefig('OT_vs_CR_withoutIS_CVresults.png')