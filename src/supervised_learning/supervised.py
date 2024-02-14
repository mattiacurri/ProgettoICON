import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.metrics import classification_report_imbalanced
import re

from plot import visualizeMetricsGraphs, plot_learning_curves
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split, RepeatedStratifiedKFold, cross_validate,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
# from sklearn_genetic import GASearchCV
# from sklearn_genetic.space import Continuous, Categorical, Integer
# from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
import matplotlib.pyplot as plt
from pprint import pprint

# lightgbm

from lightgbm import LGBMClassifier

def sturgeRule(n):
    return int(1 + 3.322 * np.log10(n))

def returnBestHyperparameters(dataset, differentialColumn):
    # Cross Validation Strategy (Repeated Stratified K-Fold) with 5 splits and 2 repeats and a random state of 42 for reproducibility
    X = dataset.drop(differentialColumn, axis=1)
    y = dataset[differentialColumn]

    # Splitting the dataset into the Training set and Test set (80% training, 20% testing) with stratification and shuffling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42
    )

    CV = RepeatedStratifiedKFold(n_splits=sturgeRule(X_train.shape[0]), n_repeats=2, random_state=42)

    # Models Evaluated
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    knn = KNeighborsClassifier()
    xgb = XGBClassifier()
    lgbm = LGBMClassifier()

    # Hyperparameters for each model
    LGBMHyperparameters = {
        "LGBM__learning_rate": [0.01, 0.10], # 0.05
        "LGBM__max_depth": [2, 5], # 3
        "LGBM__n_estimators": [50, 100], # 200
        "LGBM__lambda": [0.01, 1.0], # 0.1
        "LGBM__num_leaves": [15, 30], #31, 127
        "LGBM__min_gain_to_split": [0.1],
        "LGBM__verbose": [0],
    }
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
        "RandomForest__n_estimators": [10, 100, 200],
        "RandomForest__max_depth": [5, 10, 20],
        "RandomForest__min_samples_split": [2, 5, 10],
        "RandomForest__min_samples_leaf": [1, 2, 5],
    }
    XGBoostHyperparameters = {
        'XGBoost__learning_rate': [0.01, 0.1, 0.15],
        'XGBoost__max_depth': [2, 3, 5],
        'XGBoost__n_estimators': [20, 40],
        'XGBoost__lambda': [0.01, 0.1, 1.0]
    }

    """halvinggridsearchCV_knn = HalvingGridSearchCV(
        Pipeline([("KNearestNeighbors", knn)]),
        KNeighboursHyperparameters,
        cv=CV,
        verbose=3,
        n_jobs=-1,
    )
    SMOTENC([0, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], sampling_strategy="all", random_state=42))
    """
    '''KNeighboursHyperparameterss = {
        "n_neighbors": Integer(1, 3),
        "weights": Categorical(["uniform", "distance"]),
        "algorithm": Categorical(["ball_tree", "kd_tree"]),
        "leaf_size": Integer(1, 3),
        "p": Integer(1, 3),
    }'''

    # GridSearchCV for each model with the respective hyperparameters
    # we aim to maximize balanced_accuracy score for each model because of the imbalanced dataset
    # balanced accuracy is the arithmetic mean of sensitivity and specificity
    '''geneticSearchCV_knn = GASearchCV(
        estimator=knn,
        cv=CV,
        param_grid=KNeighboursHyperparameterss,
        scoring="balanced_accuracy",
        verbose=3,
        n_jobs=-1,
        population_size=10,
        crossover_probability=0.7,
        mutation_probability=0.3,
        algorithm="eaMuPlusLambda",
        keep_top_k=3,
        elitism=True,
        tournament_size=5,
        criteria="max",
        generations=10,
    )

    geneticSearchCV_knn.fit(X_train, y_train)

    plot_fitness_evolution(geneticSearchCV_knn)
    plt.show()

    
    print(geneticSearchCV_knn.best_params_)'''

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

    gridSearchCV_knn = GridSearchCV(
        Pipeline([("KNearestNeighbors", knn)]),
        KNeighboursHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
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
    gridSearchCV_knn.fit(X_train, y_train)
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

def trainModelKFold(dataSet, differentialColumn):
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

    bestParameters, X_train, y_train, X_test, y_test = returnBestHyperparameters(dataSet, differentialColumn)

    # save best parameters to file

    f = open("../../best_parameters.txt", "w")
    f.write(str(bestParameters))
    f.close()

    print("\033[94m")
    pprint(bestParameters)
    print("\033[0m")

    X = dataSet.drop(differentialColumn, axis=1).to_numpy()
    y = dataSet[differentialColumn].to_numpy()

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
    knn = KNeighborsClassifier(
        n_neighbors=bestParameters["KNearestNeighbors__n_neighbors"],
        weights=bestParameters["KNearestNeighbors__weights"],
        algorithm=bestParameters["KNearestNeighbors__algorithm"],
        leaf_size=bestParameters["KNearestNeighbors__leaf_size"],
        p=bestParameters["KNearestNeighbors__p"],
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
    scoring_metrics = ["balanced_accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

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
    plot_learning_curves(xgb, X, y, differentialColumn, "XGBoost", '-', cv)
    plot_learning_curves(knn, X, y, differentialColumn, "KNearestNeighbors", '-', cv)
    plot_learning_curves(dtc, X, y, differentialColumn, "DecisionTree", '-', cv)
    plot_learning_curves(rfc, X, y, differentialColumn, "RandomForest", '-', cv)
    plot_learning_curves(lgbm, X, y, differentialColumn, "LightGBM", '-', cv)

    # Visualizing the metrics for each model
    visualizeMetricsGraphs(model,"Punteggio Medio per ogni modello")
    return model, rfc, dtc, knn, xgb, lgbm, X_test, y_test, X_train, y_train


file_path = "../../data/dataset_preprocessed.csv"

df = pd.read_csv(file_path)
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
print(df.info())
differentialColumn = "Rating"

print(df["Rating"].value_counts())

# df.drop_duplicates(inplace=True)

print(df["Rating"].value_counts())

X = df.drop(columns=["Rating"])

y = df["Rating"]

model, rfc, dtc, knn, xgb, lgmb, X_test, y_test, X_train, y_train = trainModelKFold(df, differentialColumn)


lgmb.fit(X_train, y_train)
knn.fit(X_train, y_train)
dtc.fit(X_train, y_train)
rfc.fit(X_train, y_train)
xgb.fit(X_train, y_train)

y_pred_lgmb = lgmb.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_dtc = dtc.predict(X_test)
y_pred_rfc = rfc.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

# print the confusion matrix for each model

from sklearn.metrics import classification_report
# save to file

f = open("../../classification_report.txt", "w")

f.write("LightGBM Classification Report: \n")

f.write(classification_report(y_test, y_pred_lgmb))

f.write("KNearestNeighbors Classification Report: \n")

f.write(classification_report(y_test, y_pred_knn))

f.write("DecisionTree Classification Report: \n")

f.write(classification_report(y_test, y_pred_dtc))

f.write("RandomForest Classification Report: \n")

f.write(classification_report(y_test, y_pred_rfc))

f.write("XGBoost Classification Report: \n")

f.write(classification_report(y_test, y_pred_xgb))

f.close()

print("LightGBM Classification Report: ")

print(classification_report(y_test, y_pred_lgmb))

print("KNearestNeighbors Classification Report: ")

print(classification_report(y_test, y_pred_knn))

print("DecisionTree Classification Report: ")

print(classification_report(y_test, y_pred_dtc))

print("RandomForest Classification Report: ")

print(classification_report(y_test, y_pred_rfc))

print("XGBoost Classification Report: ")

print(classification_report(y_test, y_pred_xgb))

# print the imbalanced classification report

print("LightGBM Imbalanced Classification Report: ")

print(classification_report_imbalanced(y_test, y_pred_lgmb))

print("KNearestNeighbors Imbalanced Classification Report: ")

print(classification_report_imbalanced(y_test, y_pred_knn))

print("DecisionTree Imbalanced Classification Report: ")

print(classification_report_imbalanced(y_test, y_pred_dtc))

print("RandomForest Imbalanced Classification Report: ")

print(classification_report_imbalanced(y_test, y_pred_rfc))

print("XGBoost Imbalanced Classification Report: ")

print(classification_report_imbalanced(y_test, y_pred_xgb))

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

    plt.title(f"{names[i]} Feature Importances")

    # horizontal bar plot

    plt.barh(range(X_train.shape[1]), importances[indices], align="center")

    plt.yticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices])

    plt.xlabel("Relative Importance")

    # save to file

    plt.savefig(f'../../plots/feature_importances_{names[i]}.png')

    plt.show()
