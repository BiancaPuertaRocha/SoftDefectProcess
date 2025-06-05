import numpy as np
import pandas as pd
import optuna

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class RandomUnderSamplerBalancer:
    """
    Applies Random Undersampling (RUS) with hyperparameter optimization using Optuna.
    Optimizes the sampling_strategy parameter to balance the dataset.
    """
    def __init__(self, classifier, n_trials=20, direction="maximize", sampler=None, random_state=42):
        """
        Initializes the balancer with a classifier and Optuna optimization settings.

        Args:
            classifier: A scikit-learn compatible classifier.
            n_trials: Number of Optuna trials.
            direction: 'maximize' or 'minimize' based on scoring metric.
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

    def _evaluate_params(self, df, sampling_strategy,  return_data=False):
        """
        Applies Random Undersampling and evaluates classifier performance.

        Args:
            df: DataFrame with features and target column 'failure_prone'.
            sampling_strategy: Float between 0 and 1 indicating resampling ratio.

        Returns:
            Mean cross-validation ROC-AUC score.
        """
        df_clean = df.dropna()
        X = df_clean.drop(columns=['failure_prone'])
        y = df_clean['failure_prone']

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply Random Undersampling
        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_resampled, y_resampled = rus.fit_resample(X_scaled, y)

        if return_data:
            return X_resampled, y_resampled

        # Evaluate model
        score = cross_val_score(self.classifier, X_resampled, y_resampled, cv=5, scoring='roc_auc').mean()
        return score

    def _objective(self, trial):
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial object.

        Returns:
            ROC-AUC score for the given trial.
        """
        sampling_strategy = trial.suggest_float("sampling_strategy", 0.1, 0.9)

        try:
            score = self._evaluate_params(self.df, sampling_strategy)
        except Exception as e:
            print(e)
            return 0.0
        return score

    def run(self, df):
        """
        Runs the optimization and returns best parameters and score.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary with best parameters and best score.
        """
        self.df = df
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\nBest parameters found (Random Undersampling):")
        print(self.best_params)
        print(f"Mean AUC after resampling: {self.best_score:.4f}")

        return {
            'best_params': self.best_params,
            'best_cross_validation_score': self.best_score
        }
