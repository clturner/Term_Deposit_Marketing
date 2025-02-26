import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from tabulate import tabulate
from sklearn.preprocessing import PowerTransformer


def convert_binary_columns(df):
    """
    Converts binary columns in a DataFrame with "yes"/"no" or "Yes"/"No" values 
    to 0 for "no" and 1 for "yes".

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with converted binary columns.
    """
    # Iterate through columns
    for column in df.columns:
        # Check if column is of object type and has only 'yes'/'no' or 'Yes'/'No' values (case-insensitive)
        if df[column].dtype == 'object' and df[column].dropna().str.lower().isin(['yes', 'no']).all():
            print(f"Converting '{column}' column to binary values.")
            df[column] = df[column].str.strip().str.lower().map({'yes': 1, 'no': 0})
        else:
            print(f"Skipping '{column}' column (not a binary column with 'yes'/'no' values).")
    
    return df

def convert_boolean_columns(df):
    """
    Converts all Boolean columns in a DataFrame to 0 (for False) and 1 (for True).
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with Boolean columns converted to integers.
    """
    # Identify Boolean columns
    boolean_columns = df.select_dtypes(include=['bool']).columns

    # Convert Boolean columns to integers
    df[boolean_columns] = df[boolean_columns].astype(int)

    return df


import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

def interactive_categorical_encoding(df):
    """
    Interactively allows the user to choose encoding methods for each categorical column
    and modifies the DataFrame based on the user's input.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with categorical columns encoded based on user's choices.
    """
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_columns:
        print("No categorical columns found.")
        return df
    
    encoded_df = df.copy()
    
    # Store the encoding decisions
    encoding_choices = {}

    def handle_column_choice(column):
        """
        Display options for a specific column.
        """
        clear_output(wait=True)

        print(f"Configuring encoding for column: {column}")
        unique_values = encoded_df[column].unique()

        print(f"Unique values in {column}: {list(unique_values)}")

        # Dropdown to select encoding type
        encoding_type = widgets.Dropdown(
            options=['One-Hot Encoding', 'Ordinal Encoding'],
            value='One-Hot Encoding',
            description='Encoding:',
        )

        # Checkbox for one-hot encoding drop_first
        drop_first = widgets.Checkbox(
            value=False,
            description='Drop First (One-Hot Encoding)',
            disabled=False
        )

        # Text input for ordinal values
        ordinal_input = widgets.Textarea(
            value='',
            placeholder='Enter ordinal mapping as key:value pairs, e.g., Low:1, Medium:2, High:3',
            description='Ordinal Map:',
            disabled=True
        )

        # Update visibility based on encoding type
        def on_encoding_type_change(change):
            if change.new == 'One-Hot Encoding':
                drop_first.disabled = False
                ordinal_input.disabled = True
            elif change.new == 'Ordinal Encoding':
                drop_first.disabled = True
                ordinal_input.disabled = False

        encoding_type.observe(on_encoding_type_change, names='value')

        # Button to confirm choices
        confirm_button = widgets.Button(description='Confirm')

        def on_confirm_click(_):
            choice = encoding_type.value
            if choice == 'One-Hot Encoding':
                encoding_choices[column] = {'type': 'one-hot', 'drop_first': drop_first.value}
            elif choice == 'Ordinal Encoding':
                try:
                    ordinal_map = {kv.split(':')[0].strip(): int(kv.split(':')[1].strip()) 
                                   for kv in ordinal_input.value.split(',')}
                    encoding_choices[column] = {'type': 'ordinal', 'mapping': ordinal_map}
                except Exception as e:
                    print(f"Error parsing ordinal mapping: {e}")
                    return

            # Proceed to the next column
            next_column()

        confirm_button.on_click(on_confirm_click)

        display(widgets.VBox([encoding_type, drop_first, ordinal_input, confirm_button]))

    # Process columns one by one
    column_index = 0

    def next_column():
        nonlocal column_index
        if column_index < len(categorical_columns):
            column = categorical_columns[column_index]
            column_index += 1
            handle_column_choice(column)
        else:
            clear_output(wait=True)
            print("Encoding completed. Applying transformations...")
            apply_transformations()

    def apply_transformations():
        """
        Apply the selected transformations to the DataFrame.
        """
        nonlocal encoded_df
        for column, choice in encoding_choices.items():
            if choice['type'] == 'one-hot':
                encoded_df = pd.get_dummies(encoded_df, columns=[column], drop_first=choice['drop_first'])
            elif choice['type'] == 'ordinal':
                encoded_df[column] = encoded_df[column].map(choice['mapping'])

        print("Transformations applied successfully!")
        display(encoded_df.head())

    # Start the process
    next_column()

    return encoded_df



def analyze_skewness_kurtosis(df):
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=[np.number])

    # Preparing data for tabulation and dictionary
    results = []
    skew_kurt_dict = {}
    
    for column in numeric_columns.columns:
        # Get values for the column and compute skewness and kurtosis
        column_values = numeric_columns[column].dropna().to_numpy()  # Ensure no NaNs are included
        skewness = skew(column_values, bias=False)
        kurt = kurtosis(column_values, bias=False)
        # Append the results for tabulation
        results.append([column, skewness, kurt])
        # Update the dictionary with skewness and kurtosis values
        skew_kurt_dict[column] = (skewness, kurt)
    
    # Creating the tabulated DataFrame
    headers = ["Feature", "Skewness", "Kurtosis"]
    tabulated_df = tabulate(results, headers=headers, tablefmt="grid")
    
    return tabulated_df, skew_kurt_dict

import pandas as pd
import numpy as np
from scipy.stats import boxcox, yeojohnson
from scipy.stats import skew

def transform_skewed_features(df, skew_kurt_dict, skew_threshold=0.75):

    df_transformed = df.copy()
    
    for column, (skewness, _) in skew_kurt_dict.items():
        print("\n")
        column_values = df[column].dropna()  # Drop NaN values to avoid issues

        if set(column_values.unique()) <= {0, 1}:  # Check if column is binary (0/1)
            print(f"Column '{column}' is binary (Yes/No or True/False). No transformation applied.")
            continue  # Skip binary columns

        if abs(skewness) > skew_threshold:

            #column_values = df[column].dropna()  # Drop NaN values to avoid issues
            
            # Check if all values are numeric and non-negative
            if column_values.dtype.kind in 'bifc':  # Numeric types
                
                if column_values.min() < 0:  # If values are negative
                    pt = PowerTransformer(method='yeo-johnson')
                    new_column_name = f"{column}_yj_corrected"
                    #df_transformed[new_column_name] = pt.fit_transform(df_transformed[['balance']])
                    df_transformed[new_column_name] = pt.fit_transform(df_transformed[[column]])

                    # Check the skewness before and after transformation
                    original_skewness = skew(column_values)
                    transformed_skewness = skew(df_transformed[new_column_name].dropna())
                    print(f"Skewness for '{column}' before transformation: {original_skewness}")
                    print(f"Skewness for '{new_column_name}' after transformation: {transformed_skewness}")
                    
                    df_transformed.drop(columns=[column], inplace=True)
                    print(f"Dropped original column '{column}' after creating '{new_column_name}'.")
                
                elif column_values.min() == 0:  # If values include zero
                    print(f"Column '{column}' contains zero values. Applying log1p transformation.")

                    new_column_name = f"{column}_log1p_corrected"
                    #df_transformed[new_column_name] = np.log1p(df_transformed[column])

                    df_transformed[new_column_name] = (df_transformed[column] - df_transformed[column].mean()) / df_transformed[column].std()

                    # Check the skewness before and after transformation
                    original_skewness = skew(column_values)
                    transformed_skewness = skew(df_transformed[new_column_name].dropna())
                    print(f"Skewness for '{column}' before transformation: {original_skewness}")
                    print(f"Skewness for '{new_column_name}' after transformation: {transformed_skewness}")

                    df_transformed.drop(columns=[column], inplace=True)
                
                else:  # Positive values
                    new_column_name = f"{column}_bc_corrected"
                    df_transformed[new_column_name], _ = boxcox(df_transformed[column] + 1)
                    df_transformed.drop(columns=[column], inplace=True)
            else:
                print(f"Skipping non-numeric column: {column}")
        else:
            print(f"Skewness for column '{column}' is within the acceptable range. No transformation applied.")

    return df_transformed



