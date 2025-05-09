import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os

# data science libraries
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve

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

def print_highly_correlated(df, threshold=0.7) -> None:
    """Prints pairs of highly correlated columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame.
        threshold (float): The correlation threshold (absolute value). Default = 0.7
    """
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                print(f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}")
    
# ML modeling

def get_stratified_kfold(X, y, n_splits=5):
    """
    Create a Stratified K-Folds cross-validator.
    
    Parameters:
    X (pd.DataFrame): Features DataFrame.
    y (pd.Series): Target variable Series.
    n_splits (int): Number of splits for K-Folds.
    
    Returns:
    StratifiedKFold: Stratified K-Folds cross-validator object.
    """
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
    
    return skf.split(X, y)

def get_feature_importances(model, X, y):
    """
    Get feature importances from a fitted model.
    
    Parameters:
    model: Fitted model object.
    X (pd.DataFrame): Features DataFrame.
    y (pd.Series): Target variable Series.
    
    Returns:
    pd.DataFrame: DataFrame of feature importances.
    """
    
    # Fit the model
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame of feature importances
    feature_importances = pd.DataFrame(importances, index=X.columns, columns=["Importance"]).sort_values(by="Importance", ascending=False)
    
    return feature_importances

def plot_feature_importances(feature_importances, n=10):
    """
    Plot the top n feature importances.
    
    Parameters:
    feature_importances (pd.DataFrame): DataFrame of feature importances.
    n (int): Number of top features to plot.
    
    Returns:
    None
    """
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances["Importance"][:n], y=feature_importances.index[:n])
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

def save_model(model, filename):
    """
    Save the model to a file.
    
    Parameters:
    model: Model object to save.
    filename (str): Filename to save the model.
    
    Returns:
    None
    """
    
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Load the model from a file.
    
    Parameters:
    filename (str): Filename to load the model from.
    
    Returns:
    Model object: Loaded model.
    """
    
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    
    Parameters:
    model: Fitted model object.
    X_test (pd.DataFrame): Testing features DataFrame.
    y_test (pd.Series): Testing target variable Series.
    
    Returns:
    None
    """
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # # Calculate accuracy
    # accuracy = np.mean(y_pred == y_test)
    
    # print(f"Model Accuracy: {accuracy:.2%}")
    # calculate PR AUC
    print("Calculating PR AUC...")
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    print(f"Model PR AUC: {pr_auc:.2f}")
    # plot PR curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # create a confusion matrix
    print
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    # print classification report
    
    print("Classification Report:")
    # print(classification_report(y_test, y_pred, target_names=model.classes_))
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in model.classes_]))

    # calibration curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
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