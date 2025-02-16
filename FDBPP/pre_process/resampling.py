from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def balance_data_with_smote(X, y, sampling_strategy='auto', random_state=42):
    """
    Applies SMOTE to balance imbalanced data.

    Parameters:
    - X: DataFrame or array containing the features.
    - y: Array or Series with the target labels.
    - sampling_strategy: Sampling strategy (default is 'auto').
    - random_state: Seed for reproducibility.

    Returns:
    - X_resampled: Balanced features.
    - y_resampled: Balanced labels.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# For categorical data
def balance_data_with_smotenc(X, y, categorical_features, sampling_strategy='auto', random_state=42):
    """
    Balances imbalanced data using either SMOTE or SMOTENC, depending on feature types.

    Parameters:
    - X: DataFrame or array containing the features.
    - y: Array or Series with the target labels.
    - categorical_features: List of indices of categorical columns in X (for SMOTENC).
    - sampling_strategy: Sampling strategy (default is 'auto').
    - random_state: Seed for reproducibility.

    Returns:
    - X_resampled: Balanced features.
    - y_resampled: Balanced labels.
    """

    # Check for categorical columns
    if categorical_features:
        # Use SMOTENC for categorical data
        smote = SMOTENC(categorical_features=categorical_features,
                        sampling_strategy=sampling_strategy,
                        random_state=random_state)
    else:
        # Use standard SMOTE for numerical data
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

    # Apply the chosen method to balance the classes
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled


def balance_data_with_adasyn(X, y, sampling_strategy='auto', random_state=42, n_neighbors=5):
    """
    Applies ADASYN to balance imbalanced data.

    Parameters:
    - X: DataFrame or array containing the features.
    - y: Array or Series with the target labels.
    - sampling_strategy: Sampling strategy (default is 'auto').
    - random_state: Seed for reproducibility.
    - n_neighbors: Number of neighbors used to generate synthetic samples.

    Returns:
    - X_resampled: Balanced features.
    - y_resampled: Balanced labels.
    """
    # Encode non-numeric columns if any
    X_encoded = X.copy()
    
    # Encode categorical columns using LabelEncoder
    for col in X_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])

    # Apply ADASYN
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state, n_neighbors=n_neighbors)
    X_resampled, y_resampled = adasyn.fit_resample(X_encoded, y)
    
    return X_resampled, y_resampled


def balance_data_with_undersampling(X, y, sampling_strategy='auto', random_state=42):
    """
    Applies Random Under Sampling to balance imbalanced data.

    Parameters:
    - X: DataFrame or array containing the features.
    - y: Array or Series with the target labels.
    - sampling_strategy: Desired sample ratio (default is 'auto', which balances classes equally).
    - random_state: Seed for reproducibility.

    Returns:
    - X_resampled: Balanced features.
    - y_resampled: Balanced labels.
    """
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled
