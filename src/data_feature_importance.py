import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def process_feature_importance(dataframes, dataframe_names, model, X_train_dict, y_train_dict, X_test_dict, y_test_dict):
    """
    Process feature importance for each dataframe in the list using the provided model.
    
    This function iterates over the given dataframes and their corresponding names, 
    fits the provided model on the balanced training data, computes feature importances, 
    visualizes the feature importances as bar plots, and returns the feature importances 
    for each dataframe in a dictionary.
    
    Parameters:
    -----------
    dataframes : list of pandas.DataFrame
        A list of dataframes that contain the data to be used for feature importance computation.
        
    dataframe_names : list of str
        A list of strings containing the names of the corresponding dataframes. 
        These names will be used to access specific variables in the global environment.
        
    model : sklearn or xgboost estimator
        An instantiated machine learning model (e.g., XGBoost, RandomForest) that has a 
        `fit()` method and a `feature_importances_` attribute after fitting.

    X_train_dict : dict
        A dictionary containing the training data for each dataframe, where keys are the dataframe names 
        and values are the corresponding `X_train_balanced` DataFrame.

    y_train_dict : dict
        A dictionary containing the target labels for each dataframe, where keys are the dataframe names 
        and values are the corresponding `y_train_balanced` Series.

    X_test_dict : dict
        A dictionary containing the test data for each dataframe, where keys are the dataframe names 
        and values are the corresponding `X_test` DataFrame.

    y_test_dict : dict
        A dictionary containing the test labels for each dataframe, where keys are the dataframe names 
        and values are the corresponding `y_test` Series.

    Returns:
    --------
    feature_importance_results : dict
        A dictionary where keys are the names of the dataframes (from `dataframe_names`) 
        and the values are pandas DataFrames containing the feature importances for each dataframe.
        Each DataFrame contains two columns: 'Feature' and 'Importance', sorted by 'Importance' in descending order.
    """
    
    # Check if the lengths of dataframes and dataframe_names match
    if len(dataframes) != len(dataframe_names):
        raise ValueError("The number of dataframes and dataframe names must be the same length.")

    feature_importance_results = {}

    # Iterate over the dataframe names and dynamically access the corresponding data
    for df, name in zip(dataframes, dataframe_names):
        print(f"!Processing feature importance for {name}...")

        # Access the previously stored training and testing data from the passed dictionaries
        X_train_balanced = X_train_dict.get(name)  # Get training data
        y_train_balanced = y_train_dict.get(name)  # Get target data
        X_test = X_test_dict.get(name)  # Get test data (not used in training)
        y_test = y_test_dict.get(name)  # Get test labels (not used in training)

        if X_train_balanced is None or y_train_balanced is None:
            raise ValueError(f"Training data for {name} not found. Make sure the variables are correctly defined.")

        # Get model name dynamically
        model_name = str(model.__class__.__name__)

        # Fit the provided model to identify feature importance
        print(f"Fitting model for feature importance on {name}...")
        model.fit(X_train_balanced, y_train_balanced)

        # Get feature importances using the model
        feature_importances = pd.DataFrame({
            'Feature': X_train_balanced.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Store the feature importances in the results dictionary
        feature_importance_results[name] = feature_importances
        print(f"Created: {name}_{model_name}_feature_importances")

        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances)
        plt.title(f'Feature Importances from {model_name} - {name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    print(f"From function: Feature importances processed for {len(dataframes)} dataframes.")
    return feature_importance_results



def select_important_features(feature_importance_data, dataframe_names, dataframes):
    print("feature_importance_data\n", feature_importance_data)
    print("dataframe_names\n", dataframe_names)
    print("dataframes\n", dataframes)
    
    # Iterate through each dataframe name and corresponding dataframe
    for name, df in zip(dataframe_names, dataframes):

        print(f"Selecting important features for {name}...")
        print("df\n", df)

        # Get the feature importance data for the current dataframe
        importance_data = df

        # Sort by importance
        importance_data = importance_data.sort_values(by='Feature  Importance', ascending=False)
        
        # Calculate cumulative importance
        importance_data['Cumulative_Importance'] = df['Feature  Importance'].cumsum()

        # Get the features until cumulative importance is >= 0.95
        selected_features = importance_data[importance_data['Cumulative_Importance'] <= 0.95]['Feature'].tolist()
        
        print("selected_features \n", selected_features)

        # Select the columns with importance values (you can set a threshold if needed)
        #important_features = importance_data['Feature'].tolist()

