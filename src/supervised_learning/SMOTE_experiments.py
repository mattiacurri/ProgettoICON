import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from plot import visualizeMetricsGraphs, visualizeAspectRatioChart, plot_learning_curves
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split, RepeatedStratifiedKFold,
)
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder

def returnBestHyperparameters(dataset, differentialColumn):
    CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    X = dataset.drop(differentialColumn, axis=1)
    y = dataset[differentialColumn]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42
    )
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
        "RandomForest__n_estimators": [10, 100],
        "RandomForest__max_depth": [5, 10],
        "RandomForest__min_samples_split": [2, 5],
        "RandomForest__min_samples_leaf": [1, 2],
    }
    XGBoostHyperparameters = {
        'XGBoost__learning_rate': [0.1, 0.15],
        'XGBoost__max_depth': [2, 3],
        'XGBoost__n_estimators': [10, 20],
        'XGBoost__lambda': [0.1, 1.0]
    }

    # SMOTENC([0, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], sampling_strategy="all", random_state=42))
    smote = SMOTE(k_neighbors=5, random_state=42, sampling_strategy="all")
    gridSearchCV_xgb = GridSearchCV(
        Pipeline([('sampling', smote), ("XGBoost", xgb)]),
        XGBoostHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
        verbose=3
    )
    gridSearchCV_knn = GridSearchCV(
        Pipeline([('sampling', smote), ("KNearestNeighbors", knn)]),
        KNeighboursHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
        verbose=3
    )
    gridSearchCV_dtc = GridSearchCV(
        Pipeline([('sampling', smote), ("DecisionTree", dtc)]),
        DecisionTreeHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
        verbose=3
    )
    gridSearchCV_rfc = GridSearchCV(
        Pipeline([('sampling', smote), ("RandomForest", rfc)]),
        RandomForestHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
        verbose=3
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

    return bestParameters

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
    }
    bestParameters = returnBestHyperparameters(dataSet, differentialColumn)
    # print bestParameters in blue
    print("\033[94m" + str(bestParameters) + "\033[0m")
    X = dataSet.drop(differentialColumn, axis=1).to_numpy()
    y = dataSet[differentialColumn].to_numpy()

    smote = SMOTE(random_state=42, sampling_strategy="all")
    knn = Pipeline([('sampling', smote), ('KNeighborsClassifier', KNeighborsClassifier(
        n_neighbors=bestParameters["KNearestNeighbors__n_neighbors"],
        weights=bestParameters["KNearestNeighbors__weights"],
        algorithm=bestParameters["KNearestNeighbors__algorithm"],
        leaf_size=bestParameters["KNearestNeighbors__leaf_size"],
        p=bestParameters["KNearestNeighbors__p"],
        n_jobs=-1,
    ))])
    dtc = Pipeline([('sampling', smote), ('DecisionTreeClassifier', DecisionTreeClassifier(
        criterion=bestParameters["DecisionTree__criterion"],
        splitter="best",
        max_depth=bestParameters["DecisionTree__max_depth"],
        min_samples_split=bestParameters["DecisionTree__min_samples_split"],
        min_samples_leaf=bestParameters["DecisionTree__min_samples_leaf"],
    ))])
    rfc = Pipeline([('sampling', smote), ('RandomForestClassifier', RandomForestClassifier(
        n_estimators=bestParameters["RandomForest__n_estimators"],
        max_depth=bestParameters["RandomForest__max_depth"],
        min_samples_split=bestParameters["RandomForest__min_samples_split"],
        min_samples_leaf=bestParameters["RandomForest__min_samples_leaf"],
        criterion=bestParameters["RandomForest__criterion"],
        n_jobs=-1,
        random_state=42,
    ))])
    xgb = XGBClassifier(
        learning_rate=bestParameters["XGBoost__learning_rate"],
        max_depth=bestParameters["XGBoost__max_depth"],
        n_estimators=bestParameters["XGBoost__n_estimators"],
        reg_lambda=bestParameters["XGBoost__lambda"],
        n_jobs=-1,
        random_state=42,
    )
    """
    print("DecisionTree Feature Importances: ")
    for feature, importance in zip(X.columns, dtc.feature_importances_):
        print(f"\t{feature}: {importance}")

    print("RandomForest Feature Importances: ")
    for feature, importance in zip(X.columns, rfc.feature_importances_):
        print(f"\t{feature}: {importance}")
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    scoring_metrics = ["balanced_accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    results_dtc = {}
    results_rfc = {}
    results_knn = {}
    results_xgb = {}

    # apply smote to variable X

    Xs, ys = smote.fit_resample(X, y)
    # print(Xs.shape, ys.shape)
    for metric in scoring_metrics:
        scores_dtc = cross_val_score(
            dtc, Xs, ys, scoring=metric, cv=cv, n_jobs=-1
        )
        scores_rfc = cross_val_score(
            rfc, Xs, ys, scoring=metric, cv=cv, n_jobs=-1
        )
        scores_knn = cross_val_score(
            knn, Xs, ys, scoring=metric, cv=cv, n_jobs=-1
        )
        scores_xgb = cross_val_score(xgb, X, y, scoring=metric, cv=cv, n_jobs=-1)
        results_dtc[metric] = scores_dtc
        results_knn[metric] = scores_knn
        results_rfc[metric] = scores_rfc
        results_xgb[metric] = scores_xgb

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

    plot_learning_curves(xgb, X, y, differentialColumn, "XGBoost", cv)
    plot_learning_curves(knn, Xs, ys, differentialColumn, "KNearestNeighbors", cv)
    plot_learning_curves(dtc, Xs, ys, differentialColumn, "DecisionTree", cv)
    plot_learning_curves(rfc, Xs, ys, differentialColumn, "RandomForest", cv)
    mean_accuracy, mean_precision, mean_recall, mean_f1 = visualizeMetricsGraphs(model, "Punteggio medio con SMOTE")
    return model, mean_accuracy, mean_precision, mean_recall, mean_f1

# import dataset

file_path = "../../data/dataset_preprocessed.csv"

df = pd.read_csv(file_path)

differentialColumn = "Rating"

X = df.drop(columns=["Rating"])

y = df["Rating"]

modelom, mean_accuracyom, mean_precisionom, mean_recallom, mean_f1om = trainModelKFold(df, differentialColumn)
