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

# Specify the file path
file_path = "../../data/dataset.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

df_eda = pd.read_csv(file_path)

# print pearson correlation coefficient for the numeric columns
df_eda = df_eda.select_dtypes(include=["float64", "int64"])
df_eda = df_eda.drop(columns=["Binary Rating", "CIK", "SIC Code"])

df_eda = df_eda.drop(
    columns=[
        "EBIT Margin",
        "Long-term Debt / Capital",
        "Operating Margin",
        "Pre-Tax Profit Margin",
        "ROA - Return On Assets",
    ]
)

# sns.heatmap(df_eda.corr(method='pearson'), annot=True, linewidths=.5, fmt= '.1f', cmap='coolwarm')
plt.show()

# drop from df the same of df_eda
# for now I'm dropping 'Corporation', 'Ticker' and 'Rating Date'
df = df.drop(
    columns=[
        "Binary Rating",
        "EBIT Margin",
        "Long-term Debt / Capital",
        "Operating Margin",
        "Pre-Tax Profit Margin",
        "ROA - Return On Assets",
        "Corporation",
        "Ticker",
        "Rating Date",
        "SIC Code",
        "CIK",
    ]
)

# Perform one-hot encoding on Rating Agency
encoder = OneHotEncoder()
rating_agency_encoded = encoder.fit_transform(df[["Rating Agency"]]).astype(int).toarray()

# Create a new DataFrame with the encoded features
rating_agency_df = pd.DataFrame(
    rating_agency_encoded, columns=encoder.get_feature_names_out(["Rating Agency"])
)

# Concatenate the new DataFrame with the original DataFrame
df = pd.concat([df, rating_agency_df], axis=1)

# Drop the original "Rating Agency" column
df = df.drop(columns=["Rating Agency"])

# map rating
r = df["Rating"].unique()
# print(df["Rating"].value_counts())

df["Rating"] = df["Rating"].replace(
    {
        "AAA": "Rischio minimo",
        "AA+": "Rischio basso",
        "AA": "Rischio basso",
        "AA-": "Rischio basso",
        "A+": "Rischio basso",
        "A": "Rischio basso",
        "A-": "Rischio basso",
        "BBB+": "Rischio medio",
        "BBB": "Rischio medio",
        "BBB-": "Rischio medio",
        "BB+": "Rischio alto",
        "BB": "Rischio alto",
        "BB-": "Rischio alto",
        "B+": "Rischio alto",
        "B": "Rischio alto",
        "B-": "Rischio alto",
        "CCC+": "Rischio molto alto",
        "CCC": "Rischio molto alto",
        "CCC-": "Rischio molto alto",
        "CC+": "Rischio molto alto",
        "CC": "Rischio molto alto",
        "CC-": "Rischio molto alto",
        "C+": "Rischio molto alto",
        "C": "Rischio molto alto",
        "C-": "Rischio molto alto",
        "D+": "Default",
        "D": "Default",
        "D-": "Default",
    }
)

rating_dict = {
    "Rischio minimo": 0,
    "Rischio basso": 0,
    "Rischio medio": 1,
    "Rischio alto": 2,
    "Rischio molto alto": 3,
    "Default": 3,
}

# merge di Default e Rischio molto alto per mancanza di dati su default

# now map
df["Rating"] = df["Rating"].map(rating_dict)

# now map to binary
# df["Rating"] = df["Rating"].map(lambda x: 1 if x >= 6 else 0)

print(df["Rating"].value_counts())

# Perform one-hot encoding on Rating Agency
encoder = OneHotEncoder()
rating_agency_encoded = encoder.fit_transform(df[["Sector"]]).astype(int).toarray()

# Create a new DataFrame with the encoded features
rating_agency_df = pd.DataFrame(
    rating_agency_encoded, columns=encoder.get_feature_names_out(["Sector"])
)

# Concatenate the new DataFrame with the original DataFrame
df = pd.concat([df, rating_agency_df], axis=1)

# Drop the original "Sector" column
df = df.drop(columns=["Sector"])

# apply PowerTransformer() to Asset Turnover

df["Asset Turnover"] = PowerTransformer().fit_transform(df[["Asset Turnover"]])

# apply Standard Scaler to float64 columns of df

f64 = df.select_dtypes(include=["float64"])
# print(df[f64.columns].head())

scaler = MinMaxScaler()

df[f64.columns] = scaler.fit_transform(f64)


# export the new df to a csv file

df.to_csv("../../data/dataset_preprocessed.csv", index=False)