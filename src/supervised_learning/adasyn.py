from pprint import pprint

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, cohen_kappa_score

from plot import visualizeMetricsGraphs, plot_learning_curves, sturgeRule
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split, RepeatedStratifiedKFold,
)
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler, SMOTENC
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from sklearn.base import TransformerMixin, BaseEstimator
import re


class Nothing(BaseEstimator, TransformerMixin):

    def transform(self, data):
        return data

    def fit(self, data, y=None, **fit_params):
        return self


class Debugger(BaseEstimator, TransformerMixin):

    def transform(self, data):
        # Here you just print what you need + return the actual data. You're not transforming anything.

        print("Shape of Pre-processed Data:", data.shape)
        # print(pd.DataFrame(data).head())
        return data

    def fit(self, data, y=None, **fit_params):
        # No need to fit anything, because this is not an actual  transformation.

        return self


def returnBestHyperparameters(dataset, target, samplingPipe, debug=False):
    X = dataset.drop(target, axis=1)
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42
    )
    CV = RepeatedStratifiedKFold(n_splits=sturgeRule(X_train.shape[0]), n_repeats=2, random_state=42)

    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    xgb = XGBClassifier()
    lgbm = LGBMClassifier()

    LGBMHyperparameters = {
        "LGBM__learning_rate": [0.01, 0.05, 0.1],  # 0.05
        "LGBM__max_depth": [2, 5, 10],  # 3
        "LGBM__n_estimators": [50, 100, 200],  # 200
        "LGBM__lambda": [0.01, 0.1, 0.5],  # 0.1
        "LGBM__num_leaves": [5, 15],  # 31, 127
        "LGBM__min_gain_to_split": [0.1],
        "LGBM__verbose": [0],
    }
    DecisionTreeHyperparameters = {
        "DecisionTree__criterion": ["gini", "entropy", "log_loss"],
        "DecisionTree__max_depth": [5, 10, 20, 40],
        "DecisionTree__min_samples_split": [2, 5, 10, 20],
        "DecisionTree__min_samples_leaf": [2, 5, 10, 20],
        "DecisionTree__splitter": ["best"],
    }
    RandomForestHyperparameters = {
        "RandomForest__criterion": ["gini", "entropy", "log_loss"],
        "RandomForest__n_estimators": [10, 100, 200],
        "RandomForest__max_depth": [5, 10, 20],
        "RandomForest__min_samples_split": [2, 5, 10],
        "RandomForest__min_samples_leaf": [2, 5, 10],
    }
    XGBoostHyperparameters = {
        'XGBoost__learning_rate': [0.01, 0.05, 0.10],
        'XGBoost__max_depth': [5, 10, 20],
        'XGBoost__n_estimators': [20, 50, 100],
        'XGBoost__lambda': [0.01, 0.1, 0.5]
    }

    print("\033[93m")
    print("--- STARTING GRID SEARCH ---")

    gridSearchCV_lgbm = GridSearchCV(
        Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2], ("LGBM", lgbm)]),
        LGBMHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )
    gridSearchCV_xgb = GridSearchCV(
        Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2], ("XGBoost", xgb)]),
        XGBoostHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )
    gridSearchCV_dtc = GridSearchCV(
        Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2], ("DecisionTree", dtc)]),
        DecisionTreeHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )
    gridSearchCV_rfc = GridSearchCV(
        Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2], ("RandomForest", rfc)]),
        RandomForestHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )

    gridSearchCV_lgbm.fit(X_train, y_train)
    gridSearchCV_xgb.fit(X_train, y_train)
    gridSearchCV_dtc.fit(X_train, y_train)
    gridSearchCV_rfc.fit(X_train, y_train)

    bestParameters = {
        "LGBM__learning_rate": gridSearchCV_lgbm.best_params_[
            "LGBM__learning_rate"
        ],
        "LGBM__max_depth": gridSearchCV_lgbm.best_params_[
            "LGBM__max_depth"
        ],
        "LGBM__n_estimators": gridSearchCV_lgbm.best_params_[
            "LGBM__n_estimators"
        ],
        "LGBM__lambda": gridSearchCV_lgbm.best_params_[
            "LGBM__lambda"
        ],
        "LGBM__num_leaves": gridSearchCV_lgbm.best_params_[
            "LGBM__num_leaves"
        ],
        "LGBM__min_gain_to_split": gridSearchCV_lgbm.best_params_[
            "LGBM__min_gain_to_split"
        ],
        "XGBoost__learning_rate": gridSearchCV_xgb.best_params_[
            "XGBoost__learning_rate"
        ],
        "XGBoost__max_depth": gridSearchCV_xgb.best_params_[
            "XGBoost__max_depth"
        ],
        "XGBoost__n_estimators": gridSearchCV_xgb.best_params_[
            "XGBoost__n_estimators"
        ],
        "XGBoost__lambda": gridSearchCV_xgb.best_params_[
            "XGBoost__lambda"
        ],
        "DecisionTree__criterion": gridSearchCV_dtc.best_params_[
            "DecisionTree__criterion"
        ],
        "DecisionTree__max_depth": gridSearchCV_dtc.best_params_[
            "DecisionTree__max_depth"
        ],
        "DecisionTree__min_samples_split": gridSearchCV_dtc.best_params_[
            "DecisionTree__min_samples_split"
        ],
        "DecisionTree__min_samples_leaf": gridSearchCV_dtc.best_params_[
            "DecisionTree__min_samples_leaf"
        ],
        "RandomForest__n_estimators": gridSearchCV_rfc.best_params_[
            "RandomForest__n_estimators"
        ],
        "RandomForest__max_depth": gridSearchCV_rfc.best_params_[
            "RandomForest__max_depth"
        ],
        "RandomForest__min_samples_split": gridSearchCV_rfc.best_params_[
            "RandomForest__min_samples_split"
        ],
        "RandomForest__min_samples_leaf": gridSearchCV_rfc.best_params_[
            "RandomForest__min_samples_leaf"
        ],
        "RandomForest__criterion": gridSearchCV_rfc.best_params_[
            "RandomForest__criterion"
        ],
    }
    print("--- END OF GRID SEARCH ---")
    print("\033[0m")
    return bestParameters, X_train, y_train, X_test, y_test


def trainModelKFold(dataSet, target, samplingPipe, debug=False):
    print("\033[93m")
    print("--- TRAINING MODELS ---")
    if debug:
        # introduce debugging before and after samplingpipe
        print("--- DEBUG MODE ON ---")
        samplingPipe = [("DebuggerB", Debugger()), samplingPipe, ("DebuggerA", Debugger())]
    else:
        print("--- DEBUG MODE OFF ---")
        samplingPipe = [("NothingB", Nothing()), samplingPipe, ("NothingA", Nothing())]

    model = {
        "DecisionTree": {
            "balanced_accuracy": [],
            "cohen_kappa": [],
            "geometric_mean": [],
        },
        "RandomForest": {
            "balanced_accuracy": [],
            "cohen_kappa": [],
            "geometric_mean": [],
        },
        "LightGBM": {
            "balanced_accuracy": [],
            "cohen_kappa": [],
            "geometric_mean": [],
        },
        "XGBoost": {
            "balanced_accuracy": [],
            "cohen_kappa": [],
            "geometric_mean": [],
        },
    }
    print("\033[0m")
    bestParameters, X_train, y_train, X_test, y_test = returnBestHyperparameters(dataSet, target,
                                                                                 samplingPipe, debug=debug)

    f = open("adasyn_best_parameters.txt", "w")
    f.write(str(bestParameters))
    f.close()
    print("\033[94m")
    pprint(bestParameters)
    print("\033[0m")

    X = dataSet.drop(target, axis=1).to_numpy()
    y = dataSet[target].to_numpy()

    lgbm = Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2], ("LGBMClassifier", LGBMClassifier(
        learning_rate=bestParameters["LGBM__learning_rate"],
        max_depth=bestParameters["LGBM__max_depth"],
        n_estimators=bestParameters["LGBM__n_estimators"],
        reg_lambda=bestParameters["LGBM__lambda"],
        num_leaves=bestParameters["LGBM__num_leaves"],
        min_gain_to_split=bestParameters["LGBM__min_gain_to_split"],
        n_jobs=-1,
        verbose=-1,
    ))])
    dtc = Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2],
                    ('DecisionTreeClassifier', DecisionTreeClassifier(
                        criterion=bestParameters["DecisionTree__criterion"],
                        splitter="best",
                        max_depth=bestParameters["DecisionTree__max_depth"],
                        min_samples_split=bestParameters["DecisionTree__min_samples_split"],
                        min_samples_leaf=bestParameters["DecisionTree__min_samples_leaf"],
                    ))])
    rfc = Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2],
                    ('RandomForestClassifier', RandomForestClassifier(
                        n_estimators=bestParameters["RandomForest__n_estimators"],
                        max_depth=bestParameters["RandomForest__max_depth"],
                        min_samples_split=bestParameters["RandomForest__min_samples_split"],
                        min_samples_leaf=bestParameters["RandomForest__min_samples_leaf"],
                        criterion=bestParameters["RandomForest__criterion"],
                        n_jobs=-1,
                        random_state=42,
                    ))])
    xgb = Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2],
                    ("XGBoostClassifier", XGBClassifier(
                        learning_rate=bestParameters["XGBoost__learning_rate"],
                        max_depth=bestParameters["XGBoost__max_depth"],
                        n_estimators=bestParameters["XGBoost__n_estimators"],
                        reg_lambda=bestParameters["XGBoost__lambda"],
                        n_jobs=-1,
                        random_state=42,
                    ))])

    print("\033[93m")
    print("--- START EVALUATION ---")
    cv = RepeatedStratifiedKFold(n_splits=sturgeRule(X_train.shape[0]), n_repeats=2, random_state=42)
    kappa_scorer = make_scorer(cohen_kappa_score)
    geometric_mean = make_scorer(geometric_mean_score)
    scoring_metrics = ["balanced_accuracy", kappa_scorer, geometric_mean]
    results_dtc = {}
    results_rfc = {}
    results_xgb = {}
    results_lgbm = {}

    for metric in scoring_metrics:
        # Cross Validation for each model with the scoring metric and the cross validation strategy
        scores_lgbm = cross_val_score(
            lgbm, X, y, scoring=metric, cv=cv, n_jobs=-1,
        )
        scores_dtc = cross_val_score(
            dtc, X, y, scoring=metric, cv=cv, n_jobs=-1,
        )
        scores_rfc = cross_val_score(
            rfc, X, y, scoring=metric, cv=cv, n_jobs=-1
        )
        scores_xgb = cross_val_score(xgb, X, y, scoring=metric, cv=cv, n_jobs=-1)

        print("\033[94m")
        print(f"Metric: {metric}")
        print(f"DecisionTree: {scores_dtc.mean()}")
        print(f"RandomForest: {scores_rfc.mean()}")
        print(f"XGBoost: {scores_xgb.mean()}")
        print(f"LightGBM: {scores_lgbm.mean()}")
        print("\033[0m")

        results_dtc[metric] = scores_dtc
        results_rfc[metric] = scores_rfc
        results_xgb[metric] = scores_xgb
        results_lgbm[metric] = scores_lgbm

    # Storing the results for each model
    model["LightGBM"]["accuracy_list"] = results_lgbm["balanced_accuracy"]
    model["LightGBM"]["cohen_kappa"] = results_lgbm[kappa_scorer]
    model["LightGBM"]["geometric_mean"] = results_lgbm[geometric_mean]

    model["XGBoost"]["accuracy_list"] = results_xgb["balanced_accuracy"]
    model["XGBoost"]["cohen_kappa"] = results_xgb[kappa_scorer]
    model["XGBoost"]["geometric_mean"] = results_xgb[geometric_mean]

    model["DecisionTree"]["accuracy_list"] = results_dtc["balanced_accuracy"]
    model["DecisionTree"]["cohen_kappa"] = results_dtc[kappa_scorer]
    model["DecisionTree"]["geometric_mean"] = results_dtc[geometric_mean]

    model["RandomForest"]["accuracy_list"] = results_rfc["balanced_accuracy"]
    model["RandomForest"]["cohen_kappa"] = results_rfc[kappa_scorer]
    model["RandomForest"]["geometric_mean"] = results_rfc[geometric_mean]

    # Plotting the learning curves for each model
    plot_learning_curves(xgb, X, y, target, "XGBoost", 'ADASYN', cv, smote=True)
    plot_learning_curves(dtc, X, y, target, "DecisionTree", 'ADASYN', cv, smote=True)
    plot_learning_curves(rfc, X, y, target, "RandomForest", 'ADASYN', cv, smote=True)
    plot_learning_curves(lgbm, X, y, target, "LightGBM", 'ADASYN', cv, smote=True)

    # Visualizing the metrics for each model
    visualizeMetricsGraphs(model, "Punteggio Medio per ogni modello", smote=True)
    return model, rfc, dtc, xgb, lgbm, X_test, y_test, X_train, y_train



file_path = "../../data/dataset_preprocessed.csv"

df = pd.read_csv(file_path)

target = "Rating"

X = df.drop(columns=["Rating"])

y = df["Rating"]
adasyn = ADASYN(random_state=42, sampling_strategy="minority")

# ("SMOTENC", smotenc),

samplingPipes = [("ADASYN", adasyn)]

# ("SMOTEENN", smoteenn), ("SMOTETomek", smotetomek), ("ClusterCentroids", clustercentroids),
# ("ClusterCentroids", clustercentroids), ("ADASYN", adasyn)

# print(df.shape)

df = df.drop_duplicates()
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# print(df["Rating"].value_counts())
# print(df.shape)

for sPipe in samplingPipes:
    print("\033[94m")
    print("TRAINING CON:", sPipe[0])
    print("\033[0m")
    model, rfc, dtc, xgb, lgmb, X_test, y_test, X_train, y_train = trainModelKFold(df, target, sPipe,
                                                                                        False)
    models = [dtc, rfc, xgb, lgmb]
    names = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]
