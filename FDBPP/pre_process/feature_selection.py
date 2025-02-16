from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif

import numpy as np
import pandas as pd 

def fisher_score_feature_selection(X, y, threshold=0.5):
    """
    Selects features using the Fisher Score.

    Parameters:
    - X: DataFrame containing the features.
    - y: Array or Series with the target labels.
    - threshold: Minimum Fisher Score threshold for feature selection.

    Returns:
    - selected_features: List of selected feature names.
    """
    # Ensure that the index of X and y are aligned
    X_encoded = X.copy()
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
    
    # Compute the Fisher Score for each feature relative to the target
    fisher_scores, _ = f_classif(X_encoded, y)
    fisher_scores = pd.Series(fisher_scores, index=X_encoded.columns)
    
    # Select features with a Fisher Score above the threshold
    selected_features = fisher_scores[fisher_scores > threshold].index.tolist()

    return selected_features

def cfs_feature_selection(X, y, threshold=0.5, top_n=5):
    """
    Selects features using the Correlation-based Feature Selection (CFS) method.
    
    Parameters:
    - X: DataFrame containing the features.
    - y: Array or Series with the target labels.
    - threshold: Minimum correlation threshold between features and the target for selection.
    - top_n: Number of top-correlated features to be selected if none exceed the threshold.

    Returns:
    - selected_features: List of selected feature names.
    """
    X_encoded = X.copy()
    
    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])

    # Compute the correlation between features and the target variable
    corr_with_target = X_encoded.apply(lambda col: col.corr(y, method='pearson'))

    # Select features with a correlation greater than the threshold with the target
    high_corr_features = corr_with_target[abs(corr_with_target) > threshold].index.tolist()

    # If no features are selected, pick the top_n most correlated ones
    if not high_corr_features:
        high_corr_features = corr_with_target.nlargest(top_n).index.tolist()

    # Compute the correlation matrix between the selected features
    corr_matrix = X_encoded[high_corr_features].corr().abs()

    # Identify redundant features (highly correlated with each other)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    # Keep only non-redundant features
    selected_features = [feature for feature in high_corr_features if feature not in redundant_features]

    # Ensure at least one feature is selected
    if not selected_features:
        selected_features = [high_corr_features[0]]

    return selected_features

def chi_square_feature_selection(X, y, k=10):
    """
    Selects the best features using the Chi-Square Test.

    Parameters:
    - X: DataFrame containing the features (must contain only numerical and non-negative values).
    - y: Array or Series with the target labels.
    - k: Number of top features to be selected.

    Returns:
    - selected_features: List of selected feature names.
    """
    # Filter only numerical columns
    X = pd.get_dummies(X, drop_first=True)

    # Ensure values are non-negative
    if (X < 0).any().any():
        raise ValueError("The DataFrame contains negative values, which are not compatible with the Chi-Square test.")

    # Convert y into numerical labels if necessary
    if y.dtype == 'object' or isinstance(y, pd.Series):
        y = LabelEncoder().fit_transform(y)

    # Apply the Chi-Square Test to select the top k features
    chi_selector = SelectKBest(score_func=chi2, k=k)
    chi_selector.fit(X, y)

    # Retrieve the names of the selected features
    selected_features = X.columns[chi_selector.get_support()].tolist()

    return selected_features
