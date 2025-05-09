import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os

# Dataframe reading/writing

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded DataFrame.
    """
    
    df = pd.read_csv(file_path)
    print(f"Data loaded from {file_path}")
    print(f"DataFrame Shape: {df.shape}")
    print("Dataframe head")
    print(tabulate(df.head(), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".2f"))
    print("Dataframe tail")
    print(tabulate(df.tail(), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".2f"))
    
    print(f"DataFrame Columns:\n{df.columns}")
    print(f"DataFrame Info:\n{df.info()}")
    # print(f"DataFrame Description:\n{df.describe()}")
    
    return df

def save_df_to_csv(df, filepath) -> None:
    """
    Saves a Pandas DataFrame to a CSV file, creating the directory if it doesn't exist.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str): The full path to the CSV file.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(filepath, index=False)

# Dataframe overview

def get_df_head_tail_shape(df: pd.DataFrame) -> None:
    """
    Print the information of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to print its head, tail, shape.
    
    Returns:
    None
    """
    
    print("\nDataFrame Shape:")
    print(df.shape)
    print("\nDataFrame Head:")
    print(tabulate(df.head(), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".2f"))
    print("\nDataFrame Tail:")
    print(tabulate(df.tail(), headers='keys', tablefmt='pretty', showindex=False, floatfmt=".2f"))


# Dataframe subset

def get_colnames_with_prefix(df, prefix):
    """
    Get column names with a specific prefix from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to search.
    prefix (str): The prefix to search for.
    
    Returns:
    list: List of column names with the specified prefix.
    """
    
    return [col for col in df.columns if col.startswith(prefix)]

def get_colnames_with_suffix(df, suffix):
    """
    Get column names with a specific suffix from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to search.
    suffix (str): The suffix to search for.
    
    Returns:
    list: List of column names with the specified suffix.
    """
    
    return [col for col in df.columns if col.endswith(suffix)]

# ML pipeline
def execute_pipeline(df_in: pd.DataFrame, steps: list) -> pd.DataFrame:
    """
    Execute a pipeline of preprocessing steps on the DataFrame.
    
    Parameters:
    df_in (pd.DataFrame): Input DataFrame.
    steps (list): List of preprocessing steps to apply.
    
    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    
    for step in steps:
        df_in = step(df_in)
    
    return df_in

# Data split for ML

def get_X_y(df):
    """
    Split the DataFrame (train, test, holdout) into features (X) and target variable (y).
    
    Parameters:
    df (pd.DataFrame): The DataFrame to split.
    
    Returns:
    X (pd.DataFrame): Features DataFrame.
    y (pd.Series): Target variable Series.
    """
    target_var = 'In-hospital_death'
    X = df.drop(columns=[target_var]) # features
    y = df[target_var] # target
    
    return X, y

def get_train_test_holdout(X, y, test_size=0.2, holdout_size=0.2):
    """
    Split the data into train, test, and holdout sets.
    
    Parameters:
    X (pd.DataFrame): Features DataFrame.
    y (pd.Series): Target variable Series.
    test_size (float): Proportion of the data to include in the test set.
    holdout_size (float): Proportion of the data to include in the holdout set.
    
    Returns:
    X_train (pd.DataFrame): Training features DataFrame.
    X_test (pd.DataFrame): Testing features DataFrame.
    X_holdout (pd.DataFrame): Holdout features DataFrame.
    y_train (pd.Series): Training target variable Series.
    y_test (pd.Series): Testing target variable Series.
    y_holdout (pd.Series): Holdout target variable Series.
    """


    # Split into train+test and holdout
    X_train_test, X_holdout, y_train_test, y_holdout = train_test_split(X, y, test_size=holdout_size, stratify=y)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=test_size/(1-holdout_size), stratify=y_train_test)

    # Print the shapes of the splits
    print("\nTrain, Test, Holdout Split Shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"X_holdout: {X_holdout.shape}, y_holdout: {y_holdout.shape}")
    # Print the distribution of the target variable in each split
    print("\nTarget Variable Distribution:")
    print(f"y_train:\n{y_train.value_counts(normalize=True)}")
    print(f"y_test:\n{y_test.value_counts(normalize=True)}")
    print(f"y_holdout:\n{y_holdout.value_counts(normalize=True)}")
    # Print the distribution of the target variable in the entire dataset
    print(f"\nFull Dataset:\n{y.value_counts(normalize=True)}")
    
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout

# EDA plotting
def plot_missing_values(df) -> None:
    """
    Plot the percentage of missing values in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to plot missing values for.
    
    Returns:
    None
    """
    
    # Get the percentage of missing values in each column
    missing_values = df.isnull().sum() / len(df) * 100
    # Sort the missing values in descending order
    missing_values = missing_values.sort_values(ascending=False)
    
    # Plot the percentage of missing values as a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(y=missing_values.index, x=missing_values.values)
    plt.title("Percentage of Missing Values")
    plt.xlabel("Columns")
    plt.ylabel("Percentage of Missing Values")
    plt.xticks(rotation=90)
    plt.show()

    # plot count of missing values as heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

def plot_feature_nonnull_count(df) -> None:
    """
    Plot the count of non-null values in each column of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to plot non-null counts for.
    
    Returns:
    None
    """
    
    # Get the count of non-null values in each column
    nonnull_counts = df.notnull().sum()
    # Sort the non-null counts in descending order
    nonnull_counts = nonnull_counts.sort_values(ascending=True)
    
    # Plot the count of non-null values
    plt.figure(figsize=(10, 6))
    sns.barplot(y=nonnull_counts.index, x=nonnull_counts.values)
    plt.title("Count of Non-Null Values")
    plt.xlabel("Columns")
    plt.ylabel("Count of Non-Null Values")
    plt.xticks(rotation=90)
    plt.show()

def eda_on_df(df_in: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the df (with training set in mind)
    
    Parameters:
    df_in (pd.DataFrame): The training set.
    
    Returns:
    None
    """
    
    plot_missing_values(df_in)
    plot_feature_nonnull_count(df_in)

    # check for outliers using boxplot
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=df_in)
    plt.xticks(rotation=90)
    plt.title("Boxplot of features")
    plt.show()
    # check for outliers using histogram
    plt.figure(figsize=(20, 10))
    df_in.hist(bins=50, figsize=(20, 15))
    plt.xticks(rotation=90)
    plt.title("Histogram of features")
    plt.show()
    # check for correlation between features
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_in.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation heatmap of features")
    plt.show()
    

# Physionet 2012 challenge Dataset specific modules

def prep_data(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by removing unnecessary columns and handling missing values.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    
    # # Remove columns with more than 50% missing values
    # threshold = 0.5 * len(df)
    # df = df.dropna(thresh=threshold, axis=1)
    
    # # Impute missing values using KNNImputer
    # imputer = KNNImputer(n_neighbors=5)
    # df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # remove columns you don't want to use
    df_prepped = df_in.drop(columns=["SAPS-I", "SOFA","RecordID", "Survival","Length_of_stay"])
    
    return df_prepped

def clean_data(df_in) -> pd.DataFrame:
    """
    Clean the DataFrame by removing or transforming columns and handling missing values.
    
    Parameters:
    df_in (pd.DataFrame): The DataFrame to clean.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """

    df = df_in.copy() # Work on a copy from the start

    # ['ALP', 'ALT', 'AST', 'Age', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Gender', 'Glucose', 'HCO3', 'HCT', 'HR', 'Height', 'ICUType', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight', 'pH']


    # replace missing values  with NaN
    df.replace(-1.0, np.nan, inplace=True)

    # tackle general descriptor parameters - Age, Gender, Height, Weight
    df['Age'] = df['mean_Age']
    df['Gender'] = df['mean_Gender']
    df['Height'] = df['mean_Height']
    df['Weight'] = df['mean_Weight']
    # drop columns with suffix '_Age', '_Gender', '_Height', '_Weight'
    suffix_pattern = r'(_Age|_Gender|_Height|_Weight)'
    cols_to_drop_general = df.filter(regex=suffix_pattern).columns
    df.drop(columns=cols_to_drop_general, inplace=True)
    
    
    # for columns with suffix '_MechVent', replace NaN with 0 and replace with single column 'MechVent'
    mechvent_cols = [col for col in df.columns if col.endswith('_MechVent')]
    
    # Create a column 'MechVent' and fill nan values in mean_MechVent with 0 
    df['MechVent'] = df['mean_MechVent'].fillna(0)
    # drop the mechvent_cols
    df.drop(columns=mechvent_cols, inplace=True)

    # won't drop any columns yet. Will revisit this later
    
    # # Remove columns with more than 50% missing values
    # threshold = 0.5 * len(df)
    # df = df.dropna(thresh=threshold, axis=1)
    
    # # Impute missing values using KNNImputer
    # imputer = KNNImputer(n_neighbors=5)
    # df_imputed_array = imputer.fit_transform(df)
    # df = pd.DataFrame(df_imputed_array, columns=df.columns)

    return df.copy() # Return a copy of the modified DataFrame