import numpy as np
import pandas as pd
import optuna

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Chi2FeatureSelector:
    """
    Feature selection using Chi-squared test with Bayesian Optimization to find the optimal number of features (k).
    Accepts any classifier provided at initialization.
    """
    def __init__(self, classifier, n_trials=20, direction="maximize", sampler=None):
        self.classifier = classifier
        self.n_trials = n_trials
        self.direction = direction
        self.sampler = sampler
        self.study = None
        self.best_k = None
        self.best_score = None
        self.selected_features = None
        self.current_features = []  # Initialize to an empty list to avoid AttributeError

    def _evaluate_k(self, df, k):
        """
        Evaluate the performance of selecting the top k features using Chi-squared test.
        """
        X = df.drop(columns=['failure_prone'])
        y = df['failure_prone']

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Remove rows with missing values in X or y to ensure they have the same number of samples
        df_clean = df.dropna()
        X_clean = df_clean.drop(columns=['failure_prone'])
        y_clean = df_clean['failure_prone']

        # Ensure that negative values are replaced with 0
        X_clean[X_clean < 0] = 0

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Select top k features based on Chi-squared test
        selector = SelectKBest(score_func=chi2, k=k)
        X_selected = selector.fit_transform(X_scaled, y_clean)

        # Cross-validation with the provided classifier
        score = cross_val_score(self.classifier, X_selected, y_clean, cv=5, scoring='accuracy').mean()

        # Store the currently selected features
        self.current_features = X.columns[selector.get_support()].tolist()
        return score

    def _objective(self, trial):
        """
        Objective function for Optuna optimization. It suggests an optimal k and evaluates the score.

        Parameters:
        - trial: The current Optuna trial.

        Returns:
        - score: The cross-validation score for the selected number of features (k).
        """
        X = self.df.drop(columns=['failure_prone'])
        max_k = X.shape[1]
        k = trial.suggest_int("k", 1, max_k)

        try:
            score = self._evaluate_k(self.df, k)
        except Exception as e:
            print(e)
            return 0.0
        return score

    def run(self, df):
        """
        Runs the feature selection process using Optuna for hyperparameter tuning.
        
        Parameters:
        - df: The DataFrame containing the features and target column.

        Returns:
        - selected_features: The features selected after optimization.
        """
        self.df = df
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials)

        # Best number of features (k) and its corresponding score
        self.best_k = self.study.best_params["k"]
        self.best_score = self.study.best_value
        self.selected_features = self.current_features

        print(f"\nBest number of features (k): {self.best_k}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        print("Selected features:")
        print(self.selected_features)

        return self.selected_features
