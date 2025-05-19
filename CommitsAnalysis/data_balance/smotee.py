import numpy as np
import pandas as pd
import optuna

from imblearn.combine import SMOTEENN
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class SmoteeFeatureBalancer:
    """
    Applies SMOTEE (SMOTE + Edited Nearest Neighbours) with hyperparameter optimization using Optuna.
    Optimizes sampling_strategy and the number of neighbors (k_neighbors) for SMOTE.
    """
    def __init__(self, classifier, n_trials=20, direction="maximize", sampler=None, random_state=42):
        """
        Initializes the balancer with a classifier and Optuna optimization settings.

        Args:
            classifier: A scikit-learn compatible classifier.
            n_trials: Number of trials for Optuna optimization.
            direction: Optimization direction ('maximize' or 'minimize').
            sampler: Optional Optuna sampler.
            random_state: Seed for reproducibility.
        """
        self.classifier = classifier
        self.n_trials = n_trials
        self.direction = direction
        self.sampler = sampler
        self.random_state = random_state
        self.study = None
        self.best_params = None
        self.best_score = None

    def _evaluate_params(self, df, sampling_strategy, k_neighbors):
        """
        Applies SMOTEE with given parameters and evaluates classifier performance.

        Args:
            df: Input DataFrame containing features and 'failure_prone' target column.
            sampling_strategy: SMOTE sampling strategy (float between 0 and 1).
            k_neighbors: Number of neighbors for SMOTE.

        Returns:
            Mean cross-validation ROC-AUC score.
        """
        df_clean = df.dropna()
        X = df_clean.drop(columns=['failure_prone'])
        y = df_clean['failure_prone']

        # Standardize features to have zero mean and unit variance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply SMOTE + ENN
        smote_enn = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=k_neighbors
        )

        X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)

        # Evaluate classifier using cross-validation
        score = cross_val_score(self.classifier, X_resampled, y_resampled, cv=5, scoring='roc_auc').mean()
        return score

    def _objective(self, trial):
        """
        Objective function for Optuna optimization.

        Args:
            trial: An Optuna trial object.

        Returns:
            Cross-validated ROC-AUC score for the given trial's parameters.
        """
        sampling_strategy = trial.suggest_float("sampling_strategy", 0.1, 1.0)
        k_neighbors = trial.suggest_int("k_neighbors", 2, 10)

        try:
            score = self._evaluate_params(self.df, sampling_strategy, k_neighbors)
        except Exception as e:
            print(e)
            return 0.0
        return score

    def run(self, df):
        """
        Runs the SMOTEE balancing process with hyperparameter optimization.

        Args:
            df: Input DataFrame containing features and 'failure_prone' target column.

        Returns:
            Dictionary with best parameters and the best cross-validation score.
        """
        self.df = df
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\nBest parameters found (SMOTEE):")
        print(self.best_params)
        print(f"Mean AUC after resampling: {self.best_score:.4f}")

        return {
            'best_params': self.best_params,
            'best_cross_validation_score': self.best_score
        }
