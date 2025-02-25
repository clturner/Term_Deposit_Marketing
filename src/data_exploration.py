# src/data_exploration.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from IPython.display import display, Markdown
from utils import print_bold, color_value

def data_exploration(df, target):

    # Copy the DataFrame to avoid modifying the original one
    df_copy = df.copy()

    # Convert 'yes'/'no' to 1/0 in all relevant columns
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object' and set(df_copy[col].unique()) <= {'yes', 'no'}:
            df_copy[col] = df_copy[col].map({'yes': 1, 'no': 0})

    # Select numeric columns
    numeric_columns = df_copy.select_dtypes(include=['number']).columns
    print("numeric_columns: ", numeric_columns)

    # Separate binary columns before scaling
    binary_columns = [col for col in numeric_columns if df_copy[col].nunique() == 2]
    non_binary_columns = [col for col in numeric_columns if col not in binary_columns]
    print("binary_columns: \n", binary_columns)
    print("non_binary_columns: \n", non_binary_columns)

    # Scale the non-binary numeric columns
    scaler = StandardScaler()
    df_scaled = df_copy.copy()
    df_scaled[non_binary_columns] = scaler.fit_transform(df_copy[non_binary_columns])
    numeric_columns_scaled = df_scaled.select_dtypes(include=['number']).columns
    non_numeric_columns_scaled = df_scaled.select_dtypes(exclude=['number']).columns.tolist()
    #binary_columns_scaled = [col for col in numeric_columns if df_copy[col].nunique() == 2]
    #non_binary_columns_scxaled = [col for col in numeric_columns if col not in binary_columns]
                                     
    # Display the shape of the dataset
    print_bold("Shape of the dataset:")
    print(df_scaled.shape)
    print("\n")
    
    # First and Last 5 Rows
    print_bold("First 5 rows of the dataset:")
    print(df_scaled.head())
    print("\n")
    
    print_bold("Last 5 rows of the dataset:")
    print(df_scaled.tail())
    print("\n")
    
    # Display information about the dataset
    print_bold("Dataset info:")
    df_scaled.info(verbose=True)
    print("\n")
    
    # Display unique values for all columns
    print_bold("Unique values in each column:")
    for column in df_scaled.columns:
        print(f'{column}: {df[column].unique()}')
    print("\n")
    
    # Count missing values in the dataset
    print_bold("Missing values in each feature:")
    print(df_scaled.isnull().sum())
    print("\n")
    
    print_bold("Class distribution of target variable:")
    if 'y' in df_scaled.columns:
        print(df['y'].value_counts(normalize=True))  # Proportions of each class
        print("\n")

     # Check for columns with constant values
    print_bold("Columns with constant values:")
    constant_cols = [col for col in df_scaled.columns if df[col].nunique() == 1]
    if constant_cols:
        print(constant_cols)
    else:
        print("No columns in the dataset contain the same value for every row.")
    print("\n")

    # Display summary statistics
    print_bold("Summary statistics of the dataset:")
    print(df_copy.describe())
    print("\n")  
    
    # Check for missing data and print value counts
    print_bold("Missing data value counts in each column:")
    missing_data_counts = df_scaled.isnull().sum()
    print(missing_data_counts)
    print("\n")

    # Distribution of data types
    print_bold("Distribution of data types in the dataset:")
    print(df_scaled.dtypes.value_counts())
    print("\n")

    # Count non-null values in each column
    print_bold("Non-null value counts for each feature:")
    print(df_scaled.count())
    print("\n")

    # Correlation matrix
    print_bold("Correlation matrix:")
    # Filter numeric columns only
    numeric_df = df_copy.select_dtypes(include=["number"])
    if numeric_df.empty:
        print("No numeric columns available for correlation matrix.")
    else:
        # Compute the correlation matrix
        correlation_matrix = numeric_df.corr()
        print(correlation_matrix)
    print("\n")
    print("Correlation heatmap!:")
    # Increase figsize to make the plot larger and more readable
    plt.figure(figsize=(15, 12))  # Adjust size as needed
    # Create the heatmap with rotated labels
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    # Rotate the labels for better visibility
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    # Add a title
    plt.title("Correlation Heatmap")
    # Show the plot
    plt.tight_layout()  # Adjust layout to ensure nothing is cut off
    plt.show()


    # Skewness and Kurtosis analysis
    print_bold("Skewness and Kurtosis of each numeric feature Excluding categorical:")
    # Preparing data for tabulation
    results = []
    for column in numeric_columns_scaled:
        # Get values for the column and compute skewness and kurtosis
        column_values = df_scaled[column].dropna().to_numpy()  # Ensure no NaNs are included
        skewness = skew(column_values, bias=False)
        kurt = kurtosis(column_values, bias=False)
        # Append the results with the formatted color values for skewness and kurtosis
        results.append([column, color_value(skewness), color_value(kurt)])
    # Printing the results as a table
    headers = ["Feature", "Skewness", "Kurtosis"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("\n")


    print_bold("Cardinality of categorical features:")
    categorical_features = df_scaled.select_dtypes(include=['object', 'category']).columns
    for col in categorical_features:
        print(f"{col}: {df[col].nunique()} unique values")
    print("\n")

    # Plot histograms for numeric features by target variable
    print_bold("Histograms of Numeric Columns by Target Variable:")
    print(f"--- Unable to create Histogram of a Non-numeric columns:\n {non_numeric_columns_scaled}\n")
    # Calculate class counts
    
    print(df_scaled.head())
    #  Ensure the target column is binary integers
    #if df_scaled[target].dtype != "int":
     # df_scaled[target] = df_scaled[target].map({"no": 0, "yes": 1})

    class_counts = df_scaled[target].value_counts()
    for column in numeric_columns_scaled:
        if column != target:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df_scaled, x=column, hue=target, kde=True, bins=20, palette={0: "red", 1: "green"},  stat='density', common_norm=False)
            plt.title(f'Class-Proportion Normalized Histogram of {column} by {target}')
            plt.xlabel(column)
            plt.ylabel('Proportional Density')
            plt.legend(
            title=target,
            labels=['Class 1 (Yes)', 'Class 0 (No)']  # Clear labels
            )
            plt.show()


    print_bold("Pairplot of numeric features:")
    # Create a custom palette
    # Create a copy of the original DataFrame to work with
    df_pairplot = df_copy.copy()
    # get a subset of the 40000 rows
    subset_df = df_pairplot.sample(frac=0.01, random_state=42)
    # Create a custom palette where 0 is red and 1 is blue
    custom_palette = {0: "red", 1: "blue"}
    # Select numeric columns and include the target column
    numeric_columns = subset_df.select_dtypes(include=['number']).columns.tolist()
    # Create a pairplot
    sns.pairplot(subset_df, kind="kde", hue='y', diag_kind="kde", palette=custom_palette, corner=True)
    # Show the plot
    plt.show()


    print_bold("Target variable vs feature analysis:")
    for column in df_scaled.select_dtypes(include=['object', 'category']):
        if column != 'y':  # Skip target variable
            print(f"{column} vs Target Variable:")
            print(df_scaled.groupby(column)['y'].value_counts(normalize=True))
            print("\n")


    # Outlier detection using IQR
    print_bold("Outlier detection using IQR:")
    print(f"--- Unable to to detect outliers of Non-numeric columns:\n {non_numeric_columns_scaled}\n")
    print(f"--- Numeric columns with outliers:\n")
    Q1 = df_scaled[numeric_columns].quantile(0.25)
    Q3 = df_scaled[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_scaled[numeric_columns] < (Q1 - 1.5 * IQR)) | (df_scaled[numeric_columns] > (Q3 + 1.5 * IQR))).sum()
    print(outliers[outliers > 0])
    print("\n")


    # Check for outliers using boxplot
    print_bold("Box plot of the dataset and outliers:")
    plt.figure(figsize=(12, 6))  # Set the figure size
    sns.boxplot(data=df_scaled, orient='v')  # Create box plots for each feature
    plt.title("Box Plots for All Features")  # Set plot title
    plt.xlabel("Features")  # Set x-label
    plt.ylabel("Value")  # Set y-label
    plt.xticks(rotation=45)  # Rotate labels by 45 degrees
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
