from pprint import pprint

import numpy as np
import pandas as pd
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from plot import visualizeMetricsGraphs, plot_learning_curves
import matplotlib.pyplot as plt
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
from supervised import sturgeRule
from sklearn.base import TransformerMixin, BaseEstimator


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


def returnBestHyperparameters(dataset, differentialColumn, samplingPipe, debug=False):
    X = dataset.drop(differentialColumn, axis=1)
    y = dataset[differentialColumn]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42
    )
    CV = RepeatedStratifiedKFold(n_splits=sturgeRule(X_train.shape[0]), n_repeats=2, random_state=42)
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    knn = KNeighborsClassifier()
    xgb = XGBClassifier()
    KNeighboursHyperparameters = {
        "KNearestNeighbors__n_neighbors": [1, 2],
        "KNearestNeighbors__weights": ["uniform"],
        "KNearestNeighbors__algorithm": ["ball_tree"],
        "KNearestNeighbors__leaf_size": [3],
        "KNearestNeighbors__p": [3],
    }
    DecisionTreeHyperparameters = {
        "DecisionTree__criterion": ["gini", "entropy"],
        "DecisionTree__max_depth": [1, 10],
        "DecisionTree__min_samples_split": [2, 10],
        "DecisionTree__min_samples_leaf": [1, 10],
        "DecisionTree__splitter": ["best"],
    }
    RandomForestHyperparameters = {
        "RandomForest__criterion": ["gini", "entropy"],
        "RandomForest__n_estimators": [10, 50],
        "RandomForest__max_depth": [5, 10, 20],
        "RandomForest__min_samples_split": [3, 5],
        "RandomForest__min_samples_leaf": [2, 3],
    }
    XGBoostHyperparameters = {
        'XGBoost__learning_rate': [0.1, 0.15],
        'XGBoost__max_depth': [2, 3],
        'XGBoost__n_estimators': [10, 20],
        'XGBoost__lambda': [0.1, 1.0]
    }

    gridSearchCV_xgb = GridSearchCV(
        Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2], ("XGBoost", xgb)]),
        XGBoostHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",

    )
    gridSearchCV_knn = GridSearchCV(
        Pipeline(
            [samplingPipe[0], samplingPipe[1], samplingPipe[2], ("KNearestNeighbors", knn)]),
        KNeighboursHyperparameters,
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
    gridSearchCV_xgb.fit(X_train, y_train)
    gridSearchCV_knn.fit(X_train, y_train)
    gridSearchCV_dtc.fit(X_train, y_train)
    gridSearchCV_rfc.fit(X_train, y_train)
    bestParameters = {
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
        "KNearestNeighbors__n_neighbors": gridSearchCV_knn.best_params_[
            "KNearestNeighbors__n_neighbors"
        ],
        "KNearestNeighbors__weights": gridSearchCV_knn.best_params_[
            "KNearestNeighbors__weights"
        ],
        "KNearestNeighbors__algorithm": gridSearchCV_knn.best_params_[
            "KNearestNeighbors__algorithm"
        ],
        "KNearestNeighbors__leaf_size": gridSearchCV_knn.best_params_[
            "KNearestNeighbors__leaf_size"
        ],
        "KNearestNeighbors__p": gridSearchCV_knn.best_params_["KNearestNeighbors__p"],
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


def trainModelKFold(dataSet, differentialColumn, samplingPipe, debug=False):
    if debug:
        # introduce debugging before and after samplingpipe
        samplingPipe = [("DebuggerB", Debugger()), samplingPipe, ("DebuggerA", Debugger())]
    else:
        samplingPipe = [("NothingB", Nothing()), samplingPipe, ("NothingA", Nothing())]

    model = {
        "KNearestNeighbors": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        },
        "DecisionTree": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        },
        "RandomForest": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        },
        "XGBoost": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        },
        "LightGBM": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        },
    }
    bestParameters, X_train, y_train, X_test, y_test = returnBestHyperparameters(dataSet, differentialColumn,
                                                                                 samplingPipe, debug=debug)

    print("\033[94m")
    pprint(bestParameters)
    print("\033[0m")

    X = dataSet.drop(differentialColumn, axis=1).to_numpy()
    y = dataSet[differentialColumn].to_numpy()

    lgbm = Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2], ("LGBMClassifier", LGBMClassifier(
        learning_rate=bestParameters["LGBM__learning_rate"],
        max_depth=bestParameters["LGBM__max_depth"],
        n_estimators=bestParameters["LGBM__n_estimators"],
        reg_lambda=bestParameters["LGBM__lambda"],
        num_leaves=bestParameters["LGBM__num_leaves"],
        min_gain_to_split=bestParameters["LGBM__min_gain_to_split"],
        verbose=bestParameters["LGBM__verbose"],
        n_jobs=-1,
    ))])
    knn = Pipeline([samplingPipe[0], samplingPipe[1], samplingPipe[2],
                    ('KNeighborsClassifier', KNeighborsClassifier(
                        n_neighbors=bestParameters["KNearestNeighbors__n_neighbors"],
                        weights=bestParameters["KNearestNeighbors__weights"],
                        algorithm=bestParameters["KNearestNeighbors__algorithm"],
                        leaf_size=bestParameters["KNearestNeighbors__leaf_size"],
                        p=bestParameters["KNearestNeighbors__p"],
                        n_jobs=-1,
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

    cv = RepeatedStratifiedKFold(n_splits=sturgeRule(X_train.shape[0]), n_repeats=2, random_state=42)
    scoring_metrics = ["balanced_accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    print(X.shape, y.shape)
    Xs, ys = samplingPipe[1][1].fit_resample(X, y)
    print(Xs.shape, ys.shape)
    results_dtc = {}
    results_rfc = {}
    results_knn = {}
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
        scores_knn = cross_val_score(
            knn, X, y, scoring=metric, cv=cv, n_jobs=-1
        )
        scores_xgb = cross_val_score(xgb, X, y, scoring=metric, cv=cv, n_jobs=-1)

        print("\033[94m")
        print(f"Metric: {metric}")
        print(f"DecisionTree: {scores_dtc.mean()}")
        print(f"RandomForest: {scores_rfc.mean()}")
        print(f"KNearestNeighbors: {scores_knn.mean()}")
        print(f"XGBoost: {scores_xgb.mean()}")
        print(f"LightGBM: {scores_lgbm.mean()}")
        print("\033[0m")
        results_dtc[metric] = scores_dtc
        results_knn[metric] = scores_knn
        results_rfc[metric] = scores_rfc
        results_xgb[metric] = scores_xgb
        results_lgbm[metric] = scores_lgbm

    # Storing the results for each model
    model["LightGBM"]["accuracy_list"] = results_lgbm["balanced_accuracy"]
    model["LightGBM"]["precision_list"] = results_lgbm["precision_weighted"]
    model["LightGBM"]["recall_list"] = results_lgbm["recall_weighted"]
    model["LightGBM"]["f1"] = results_lgbm["f1_weighted"]

    model["XGBoost"]["accuracy_list"] = results_xgb["balanced_accuracy"]
    model["XGBoost"]["precision_list"] = results_xgb["precision_weighted"]
    model["XGBoost"]["recall_list"] = results_xgb["recall_weighted"]
    model["XGBoost"]["f1"] = results_xgb["f1_weighted"]

    model["DecisionTree"]["accuracy_list"] = results_dtc["balanced_accuracy"]
    model["DecisionTree"]["precision_list"] = results_dtc["precision_weighted"]
    model["DecisionTree"]["recall_list"] = results_dtc["recall_weighted"]
    model["DecisionTree"]["f1"] = results_dtc["f1_weighted"]

    model["RandomForest"]["accuracy_list"] = results_rfc["balanced_accuracy"]
    model["RandomForest"]["precision_list"] = results_rfc["precision_weighted"]
    model["RandomForest"]["recall_list"] = results_rfc["recall_weighted"]
    model["RandomForest"]["f1"] = results_rfc["f1_weighted"]

    model["KNearestNeighbors"]["accuracy_list"] = results_knn["balanced_accuracy"]
    model["KNearestNeighbors"]["precision_list"] = results_knn["precision_weighted"]
    model["KNearestNeighbors"]["recall_list"] = results_knn["recall_weighted"]
    model["KNearestNeighbors"]["f1"] = results_knn["f1_weighted"]

    # Plotting the learning curves for each model
    plot_learning_curves(xgb, X, y, differentialColumn, "XGBoost", samplingPipe[1][0], cv)
    plot_learning_curves(knn, X, y, differentialColumn, "KNearestNeighbors", samplingPipe[1][0], cv)
    plot_learning_curves(dtc, X, y, differentialColumn, "DecisionTree", samplingPipe[1][0], cv)
    plot_learning_curves(rfc, X, y, differentialColumn, "RandomForest", samplingPipe[1][0], cv)
    plot_learning_curves(lgbm, X, y, differentialColumn, "LightGBM", samplingPipe[1][0], cv)

    # Visualizing the metrics for each model
    visualizeMetricsGraphs(model, "Punteggio Medio per ogni modello con " + str(samplingPipe[0][0]))
    return model, rfc, dtc, knn, xgb, lgbm, X_test, y_test, X_train, y_train


# import dataset

file_path = "../../data/dataset_preprocessed.csv"

df = pd.read_csv(file_path)

differentialColumn = "Rating"

X = df.drop(columns=["Rating"])

y = df["Rating"]

# -- OVER SAMPLERS -- #
random_over_sampler = RandomOverSampler(sampling_strategy="all", random_state=42)

# smotenc = SMOTENC([0, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], sampling_strategy="all",
# random_state=42)
smote = SMOTE(random_state=42, sampling_strategy="all")

# -- UNDER AND OVER SAMPLERS -- #
smoteenn = SMOTEENN(random_state=42, sampling_strategy="all")
smotetomek = SMOTETomek(random_state=42, sampling_strategy="all")

# -- UNDERSAMPLERS -- #
random_under_sampler = RandomUnderSampler(random_state=42, sampling_strategy="all")
clustercentroids = ClusterCentroids(random_state=42, sampling_strategy="all")

# -- ADAPTIVE SYNTHETIC SAMPLERS -- #
adasyn = ADASYN(random_state=42, sampling_strategy="all")

# ("SMOTENC", smotenc),
samplingPipes = [("SMOTE", smote)]

# ("SMOTEENN", smoteenn), ("SMOTETomek", smotetomek), ("ClusterCentroids", clustercentroids), ("RandomOverSampler", random_over_sampler), ("RandomUnderSampler", random_under_sampler),
# ("ClusterCentroids", clustercentroids), ("ADASYN", adasyn)

# print(df.shape)

df = df.drop_duplicates()
print(df["Rating"].value_counts())
# print(df.shape)

for sPipe in samplingPipes:
    print("\033[94m")
    print("TRAINING CON:", sPipe[0])
    print("\033[0m")
    model, rfc, dtc, knn, xgb, lgmb, X_test, y_test, X_train, y_train = trainModelKFold(df, differentialColumn, sPipe,
                                                                                        False)

    y_pred_lgmb = lgmb.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    y_pred_dtc = dtc.predict(X_test)
    y_pred_rfc = rfc.predict(X_test)
    y_pred_xgb = xgb.predict(X_test)

    # print the confusion matrix for each model

    from sklearn.metrics import classification_report

    # save to file

    f = open("../../classification_report.txt", "w")

    f.write(f"LightGBM {sPipe[0]} Classification Report: \n")

    f.write(classification_report(y_test, y_pred_lgmb))

    f.write(f"KNN {sPipe[0]} Classification Report: \n")

    f.write(classification_report(y_test, y_pred_knn))

    f.write(f"DecisionTree {sPipe[0]} Classification Report: \n")

    f.write(classification_report(y_test, y_pred_dtc))

    f.write(f"RandomForest {sPipe[0]} Classification Report: \n")

    f.write(classification_report(y_test, y_pred_rfc))

    f.write(f"XGBoost {sPipe[0]} Classification Report: \n")

    f.write(classification_report(y_test, y_pred_xgb))

    f.close()

    from sklearn.metrics import ConfusionMatrixDisplay

    # plot the confusion matrix for each model

    ConfusionMatrixDisplay.from_estimator(lgmb, X_test, y_test, display_labels=lgmb.classes_, cmap="summer")
    plt.title("LightGBM Confusion Matrix")

    # sae to file

    plt.savefig(f'../../plots/confusion_matrix_LightGBM.png')
    plt.show()

    ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test, display_labels=knn.classes_, cmap="summer")
    plt.title("KNN Confusion Matrix")
    plt.savefig(f'../../plots/confusion_matrix_KNN.png')
    plt.show()

    ConfusionMatrixDisplay.from_estimator(dtc, X_test, y_test, display_labels=dtc.classes_, cmap="summer")
    plt.title("Decision Tree Confusion Matrix")
    plt.savefig(f'../../plots/confusion_matrix_DecisionTree.png')
    plt.show()

    ConfusionMatrixDisplay.from_estimator(rfc, X_test, y_test, display_labels=rfc.classes_, cmap="summer")
    plt.title("RandomForest Confusion Matrix")
    plt.savefig(f'../../plots/confusion_matrix_RandomForest.png')
    plt.show()

    ConfusionMatrixDisplay.from_estimator(xgb, X_test, y_test, display_labels=xgb.classes_, cmap="summer")
    plt.title("XGBoost Confusion Matrix")
    plt.savefig(f'../../plots/confusion_matrix_XGBoost.png')
    plt.show()

    # plot the feature importances for each model

    models = [dtc, rfc, xgb, lgmb]
    names = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]

    for i, m in enumerate(models):
        m.fit(X_train, y_train)

        importances = m.feature_importances_

        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(15, 15))

        plt.title(f"{names[i]} Feature Importances for {sPipe[0]}")

        # horizontal bar plot

        plt.barh(range(X_train.shape[1]), importances[indices], align="center")

        plt.yticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices])

        plt.xlabel("Relative Importance")

        # save to file

        plt.savefig(f'../../plots/feature_importances_{names[i]}_{sPipe[0]}.png')

        plt.show()
