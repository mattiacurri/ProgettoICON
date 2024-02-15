import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, KBinsDiscretizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns


file_path = "../../data/dataset.csv"
def generate_dataset_for_supervised():
    # Specify the file path
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    print(df.shape)
    print(df.info())
    df_eda = pd.read_csv(file_path)
    

    df_eda = df_eda.select_dtypes(include=["float64", "int64"])

    df_eda = df_eda.drop(
        columns=[
            "EBIT Margin",
            "Long-term Debt / Capital",
            "Operating Margin",
            "Pre-Tax Profit Margin",
            "ROA - Return On Assets","Binary Rating", "CIK", "SIC Code"
        ]
    )

    # bigger plot
    plt.figure(figsize=(15, 15))
    sns.heatmap(df_eda.corr(method='pearson'), annot=True, linewidths=.1, fmt= '.1f', cmap='coolwarm')
    plt.show()

    df = df.drop(
        columns=[
            "Binary Rating",
            "EBIT Margin",
            "Long-term Debt / Capital",
            "Operating Margin",
            "Pre-Tax Profit Margin",
            "ROA - Return On Assets",
            "SIC Code",
            "CIK",
        ]
    )
    #print(df.shape)
    #print(df.drop_duplicates().shape)
    '''
    # map ticker to numbers
    
    r = df["Ticker"].unique()
    
    df["Ticker"] = df["Ticker"].map({r[i]: i for i in range(len(r))})
    
    # map corporation to numbers
    
    r = df["Corporation"].unique()
    
    df["Corporation"] = df["Corporation"].map({r[i]: i for i in range(len(r))})
    '''

    # drop ticker
    df = df.drop(columns=["Ticker"])
    #print(df.shape)
    #print(df.drop_duplicates().shape)
    #print(df["Corporation"].value_counts())
    
    df = df.drop(columns=["Corporation"])
    # map date to year
    #print(df.shape)
    #print(df.drop_duplicates().shape)
    df["Rating Date"] = pd.to_datetime(df["Rating Date"])
    
    # new feature year AND month
    
    df["Year Month Day"] = df["Rating Date"].dt.to_period("D")
    
    df["Year Month Day"] = df["Year Month Day"].dt.to_timestamp().dt.strftime('%Y%m%d').astype(int)
    
    print(df["Year Month Day"].value_counts())
    # drop Rating Date
    
    df = df.drop(columns=["Rating Date"])
    #print(df.shape)
    #print(df.drop_duplicates().shape)
    
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

    
    #print(df["Rating"].value_counts())
    
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

    # transform rating to int32

    df["Rating"] = df["Rating"].astype("int32")
    df["Asset Turnover"] = PowerTransformer().fit_transform(df[["Asset Turnover"]])

    # apply Standard Scaler to float64 columns of df

    f64 = df.select_dtypes(include=["float64"])
    # print(df[f64.columns].head())


    scaler = MinMaxScaler()
    
    df[f64.columns] = scaler.fit_transform(f64)
    
    print(df.info())
    
    print(df["Rating"].value_counts())
    print(df.drop_duplicates().shape)
    
    df.drop_duplicates()
    # export the new df to a csv file
    print(df.shape)
    # df.to_csv("../../data/dataset_preprocessed.csv", index=False)

def prepare_dataset_for_bn():
    df = pd.read_csv(file_path)
    # print pearson correlation coefficient for the numeric columns
    df_num = df.select_dtypes(include=["float64", "int64"])
    pearson_corr = df_num.corr(method="pearson")
    # plot pearson_corr
    plt.figure(figsize=(20, 20))
    sns.heatmap(pearson_corr, annot=True, linewidths=.5, fmt='.1f', cmap='coolwarm')
    plt.show()

    df = df.drop(
        columns=[
            "Binary Rating",
            "EBIT Margin",
            "Long-term Debt / Capital",
            "Operating Margin",
            "Pre-Tax Profit Margin",
            "ROA - Return On Assets",
            "SIC Code",
            "CIK",
            "Ticker",
            "Corporation"
        ]
    )
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

    df["Rating Date"] = pd.to_datetime(df["Rating Date"])

    # new feature year AND month

    df["Year Month Day"] = df["Rating Date"].dt.to_period("D")

    df["Year Month Day"] = df["Year Month Day"].dt.to_timestamp().dt.strftime('%Y%m%d').astype(int)
    df["Rating"] = df["Rating"].astype("int16")
    df["Asset Turnover"] = PowerTransformer().fit_transform(df[["Asset Turnover"]])

    # apply Standard Scaler to float64 columns of df

    f64 = df.select_dtypes(include=["float64"])
    # print(df[f64.columns].head())

    scaler = MinMaxScaler()

    df[f64.columns] = scaler.fit_transform(f64)

    df = df.drop(columns=["Rating Date"])
    #print(df.info())
    #print(df.shape)
    #print(df.drop_duplicates().shape)

    # trying drop
    df = df.drop(columns=["Year Month Day"])
    # df = df.drop(columns=["Rating Agency", "Sector"])
    df = df.drop(columns=["Free Cash Flow Per Share", "Return On Tangible Equity", "ROE - Return On Equity"])

    discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')
    continuous_columns = df.select_dtypes(include=['float64', 'int64', 'int32']).columns
    df[continuous_columns] = discretizer.fit_transform(df[continuous_columns])
    df = df.drop_duplicates()
    print(df.info())
    print(df.shape)

    # remap rating agency

    #df.to_csv("../../data/dataset_preprocessed_bayesian.csv", index=False)

generate_dataset_for_supervised()
prepare_dataset_for_bn()

