import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN

from data_balance.whale_optimizer import WhaleOptimizer

class ADASYNBalancer:
    """
    Applies ADASYN using Whale Optimization Algorithm (WOA) for hyperparameter optimization.
    Optimizes sampling_strategy and k_neighbors to improve classifier performance.
    """
    def __init__(self, classifier, n_whales=10, n_iterations=20, random_state=42):
        self.classifier = classifier
        self.n_whales = n_whales
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.best_score = -np.inf
        self.best_params = {}

    def _evaluate_params(self, df, sampling_strategy, k_neighbors, return_data=False):
        """
        Evaluates classifier performance using ADASYN oversampling with given parameters.

        Args:
            df: Input DataFrame containing features and target column 'failure_prone'.
            sampling_strategy: Float between 0.1 and 1.0, proportion of minority class after resampling.
            k_neighbors: Number of neighbors to use in ADASYN.

        Returns:
            Mean ROC-AUC score from 5-fold cross-validation.
        """
        df_clean = df.dropna()
        X = df_clean.drop(columns=['failure_prone'])
        y = df_clean['failure_prone']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ada = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=int(k_neighbors),
            random_state=self.random_state
        )

        X_resampled, y_resampled = ada.fit_resample(X_scaled, y)
        if return_data:
            return X_resampled, y_resampled

        try:
            score = cross_val_score(self.classifier, X_resampled, y_resampled, cv=5, scoring='roc_auc').mean()
        except Exception:
            score = 0.0
        

        return score

    def run(self, df):
        """
        Runs the Whale Optimization Algorithm to find the best ADASYN hyperparameters.

        Args:
            df: Input DataFrame with features and target.

        Returns:
            Dictionary with best parameters and best cross-validation ROC-AUC score.
        """
        # Objective function for WOA optimization
        def objective(position):
            sampling_strategy = float(position[0])
            k_neighbors = int(np.round(position[1]))
            # Ensure k_neighbors stays within bounds
            k_neighbors = np.clip(k_neighbors, 2, 10)
            try:
                return self._evaluate_params(df, sampling_strategy, k_neighbors)
            except Exception as e:
                print(f"Error during evaluation: {e}")
                return 0.0

        # Parameter bounds: sampling_strategy and k_neighbors
        bounds = [(0.1, 1.0), (2, 10)]

        # Assuming WhaleOptimizer is already implemented and imported
        optimizer = WhaleOptimizer(
            objective_func=objective,
            bounds=bounds,
            n_whales=self.n_whales,
            n_iterations=self.n_iterations,
            seed=self.random_state
        )

        best_position, best_score = optimizer.optimize()

        self.best_params = {
            'sampling_strategy': float(best_position[0]),
            'k_neighbors': int(np.round(best_position[1]))
        }
        self.best_score = best_score

        print("\nBest parameters (WOA):")
        print(self.best_params)
        print(f"Mean AUC: {self.best_score:.4f}")


        return {
            'best_params': self.best_params,
            'best_cross_validation_score': self.best_score
        }