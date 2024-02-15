import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
import re
from plot import visualizeMetricsGraphs, plot_learning_curves, sturgeRule
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split, RepeatedStratifiedKFold
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from pprint import pprint
from lightgbm import LGBMClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer


def returnBestHyperparameters(dataset, target):
    # Cross Validation Strategy (Repeated Stratified K-Fold) with 5 splits and 2 repeats and a random state of 42 for reproducibility
    X = dataset.drop(target, axis=1)
    y = dataset[target]

    # Splitting the dataset into the Training set and Test set (80% training, 20% testing) with stratification and shuffling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42
    )

    CV = RepeatedStratifiedKFold(n_splits=sturgeRule(X_train.shape[0]), n_repeats=2, random_state=42)

    # Models Evaluated
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    xgb = XGBClassifier()
    lgbm = LGBMClassifier()

    # Hyperparameters for each model
    LGBMHyperparameters = {
        "LGBM__learning_rate": [0.01, 0.05, 0.1],
        "LGBM__max_depth": [2, 5, 10],
        "LGBM__n_estimators": [50, 100, 200],
        "LGBM__lambda": [0.01, 0.1, 0.5],
        "LGBM__num_leaves": [5, 15],
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

    # GridSearchCV for each model with the respective hyperparameters
    gridSearchCV_lgbm = GridSearchCV(
        Pipeline([("LGBM", lgbm)]),
        LGBMHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )

    gridSearchCV_xgb = GridSearchCV(
        Pipeline([("XGBoost", xgb)]),
        XGBoostHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy"
    )

    gridSearchCV_dtc = GridSearchCV(
        Pipeline([("DecisionTree", dtc)]),
        DecisionTreeHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )

    gridSearchCV_rfc = GridSearchCV(
        Pipeline([("RandomForest", rfc)]),
        RandomForestHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )

    # Fitting the models with the training data
    gridSearchCV_lgbm.fit(X_train, y_train)
    gridSearchCV_xgb.fit(X_train, y_train)
    gridSearchCV_dtc.fit(X_train, y_train)
    gridSearchCV_rfc.fit(X_train, y_train)

    # Returning the best hyperparameters for each model
    bestParameters = {
        "LGBM__learning_rate": gridSearchCV_lgbm.best_params_["LGBM__learning_rate"],
        "LGBM__max_depth": gridSearchCV_lgbm.best_params_["LGBM__max_depth"],
        "LGBM__n_estimators": gridSearchCV_lgbm.best_params_["LGBM__n_estimators"],
        "LGBM__lambda": gridSearchCV_lgbm.best_params_["LGBM__lambda"],
        "LGBM__num_leaves": gridSearchCV_lgbm.best_params_["LGBM__num_leaves"],
        "LGBM__min_gain_to_split": gridSearchCV_lgbm.best_params_["LGBM__min_gain_to_split"],
        "LGBM__verbose": gridSearchCV_lgbm.best_params_["LGBM__verbose"],
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
    return bestParameters, X_train, y_train, X_test, y_test


def trainModelKFold(dataSet, target):
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

    bestParameters, X_train, y_train, X_test, y_test = returnBestHyperparameters(dataSet, target)

    # save best parameters to file

    f = open("best_parameters_normal.txt", "w")
    f.write(str(bestParameters))
    f.close()

    print("\033[94m")
    pprint(bestParameters)
    print("\033[0m")

    X = dataSet.drop(target, axis=1).to_numpy()
    y = dataSet[target].to_numpy()

    # Each model is initialized with the best hyperparameters found
    lgbm = LGBMClassifier(
        learning_rate=bestParameters["LGBM__learning_rate"],
        max_depth=bestParameters["LGBM__max_depth"],
        n_estimators=bestParameters["LGBM__n_estimators"],
        reg_lambda=bestParameters["LGBM__lambda"],
        num_leaves=bestParameters["LGBM__num_leaves"],
        min_gain_to_split=bestParameters["LGBM__min_gain_to_split"],
        verbose=bestParameters["LGBM__verbose"],
        n_jobs=-1,
    )
    dtc = DecisionTreeClassifier(
        criterion=bestParameters["DecisionTree__criterion"],
        splitter="best",
        max_depth=bestParameters["DecisionTree__max_depth"],
        min_samples_split=bestParameters["DecisionTree__min_samples_split"],
        min_samples_leaf=bestParameters["DecisionTree__min_samples_leaf"],
    )
    rfc = RandomForestClassifier(
        n_estimators=bestParameters["RandomForest__n_estimators"],
        max_depth=bestParameters["RandomForest__max_depth"],
        min_samples_split=bestParameters["RandomForest__min_samples_split"],
        min_samples_leaf=bestParameters["RandomForest__min_samples_leaf"],
        criterion=bestParameters["RandomForest__criterion"],
        n_jobs=-1,
        random_state=42,
    )
    xgb = XGBClassifier(
        learning_rate=bestParameters["XGBoost__learning_rate"],
        max_depth=bestParameters["XGBoost__max_depth"],
        n_estimators=bestParameters["XGBoost__n_estimators"],
        reg_lambda=bestParameters["XGBoost__lambda"],
        n_jobs=-1,
        random_state=42,
    )
    # Cross Validation Strategy (Repeated Stratified K-Fold) with 5 splits and 2 repeats and a random state of 42 for reproducibility
    cv = RepeatedStratifiedKFold(n_splits=sturgeRule(X_train.shape[0]), n_repeats=2, random_state=42)

    # Scoring Metrics for the models (Balanced Accuracy, Precision, Recall, F1)
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
    plot_learning_curves(rfc, X, y, target, "RandomForest", '-', cv)
    plot_learning_curves(dtc, X, y, target, "DecisionTree", '-', cv)
    plot_learning_curves(lgbm, X, y, target, "LightGBM", '-', cv)
    plot_learning_curves(xgb, X, y, target, "XGBoost", '-', cv)

    # Visualizing the metrics for each model
    visualizeMetricsGraphs(model, "Punteggio Medio per ogni modello")
    return model, rfc, dtc, xgb, lgbm, X_test, y_test, X_train, y_train


file_path = "../../data/dataset_preprocessed.csv"

df = pd.read_csv(file_path)
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)) # for LightGBM

target = "Rating"

X = df.drop(columns=["Rating"])

y = df["Rating"]

df = df.drop_duplicates()

model, rfc, dtc, xgb, lgmb, X_test, y_test, X_train, y_train = trainModelKFold(df, target)

# plot the feature importances for each model

models = [dtc, rfc, xgb, lgmb]
names = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]

for i, m in enumerate(models):
    m.fit(X_train, y_train)
    importances = m.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(15, 15))
    plt.title(f"{names[i]} Feature Importances")
    plt.barh(range(X_train.shape[1]), importances[indices], align="center")
    plt.yticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.savefig(f'../../plots/feature_importances_{names[i]}.png')
    plt.show()
