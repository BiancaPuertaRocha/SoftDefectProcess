import numpy as np
import pandas as pd
import optuna

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler


class Chi2FeatureSelector:
    """
    Feature selection using the Chi-squared statistical test with Bayesian optimization (Optuna)
    to find the optimal number of features (k) that maximize model performance.
    """

    def __init__(self, classifier, n_trials=20, direction="maximize", sampler=None, min_features=3):
        """
        Initializes the feature selector.

        Args:
            classifier: A scikit-learn compatible classifier used for evaluation.
            n_trials: Number of Optuna trials for hyperparameter search.
            direction: 'maximize' or 'minimize' for Optuna study direction.
            sampler: Optional Optuna sampler to control the sampling behavior.
            min_features: Minimum number of features to consider during search.
        """
        self.classifier = classifier
        self.n_trials = n_trials
        self.direction = direction
        self.sampler = sampler
        self.study = None
        self.best_k = None
        self.best_score = None
        self.selected_features = None
        self.current_features = []  # Stores features selected in the last evaluated trial
        self.min_features = min_features

    def _evaluate_k(self, df, k):
        """
        Evaluates model performance using the top-k features selected by the Chi-squared test.

        Args:
            df: A pandas DataFrame containing features and a target column named 'failure_prone'.
            k: The number of top features to select.

        Returns:
            Mean ROC-AUC score from cross-validation using the selected features.
        """
        # Remove rows with missing values to ensure alignment between X and y
        df_clean = df.dropna()

        # Split into features and target
        X_clean = df_clean.drop(columns=['failure_prone'])
        y_clean = df_clean['failure_prone']

        # Scale features to the [0, 1] range as required by the chi-squared test
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Select top-k features based on Chi-squared statistics
        selector = SelectKBest(score_func=chi2, k=k)
        X_selected = selector.fit_transform(X_scaled, y_clean)

        # Evaluate performance using 5-fold cross-validation
        score = cross_val_score(self.classifier, X_selected, y_clean, cv=5, scoring='balanced_accuracy').mean()

        # Save the feature names selected in this trial
        self.current_features = X_clean.columns[selector.get_support()].tolist()
        return score

    def _objective(self, trial):
        """
        Optuna objective function that suggests and evaluates a value for k.

        Args:
            trial: An Optuna trial object used for sampling.

        Returns:
            Cross-validation ROC-AUC score using the selected features.
        """
        # Determine the number of features available
        X = self.df.drop(columns=['failure_prone'])
        max_k = X.shape[1]

        # Suggest a value for k within the valid range
        k = trial.suggest_int("k", self.min_features, max_k)

        try:
            score = self._evaluate_k(self.df, k)
        except Exception as e:
            print(e)
            return 0.0  # Return a low score if something goes wrong
        return score

    def run(self, df):
        """
        Starts the feature selection process using Optuna to tune k.

        Args:
            df: A DataFrame containing features and a target column named 'failure_prone'.

        Returns:
            A dictionary with the selected features, best k, and best cross-validation score.
        """
        self.df = df

        # Create the Optuna study for Bayesian optimization
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials)

        # Extract best results
        self.best_k = self.study.best_params["k"]
        self.best_score = self.study.best_value
        self.selected_features = self.current_features

        # Display best results
        print(f"\nBest number of features (k): {self.best_k}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        print("Selected features:")
        print(self.selected_features)

        return {
            'features': self.selected_features,
            'best_k': self.best_k,
            'best_cross_validation_score': self.best_score
        }
