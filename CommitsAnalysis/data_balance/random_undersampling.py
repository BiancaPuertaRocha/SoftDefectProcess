import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from data_balance.whale_optimizer import WhaleOptimizer

class RandomUnderSamplerBalancer:
    """
    Applies Random Undersampling (RUS) with hyperparameter optimization using Whale Optimization Algorithm (WOA).
    Optimizes the sampling_strategy parameter to balance the dataset.
    """
    def __init__(self, classifier, n_whales=10, n_iterations=20, random_state=42):
        self.classifier = classifier
        self.n_whales = n_whales
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.best_score = -1.0
        self.best_params = {}

    def _evaluate_params(self, df, sampling_strategy, return_data=False):
        df_clean = df.dropna()
        X = df_clean.drop(columns=['failure_prone'])
        y = df_clean['failure_prone']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
        X_resampled, y_resampled = rus.fit_resample(X_scaled, y)
        
        if return_data:
            return X_resampled, y_resampled

        score = cross_val_score(self.classifier, X_resampled, y_resampled, cv=5, scoring='roc_auc').mean()
        return score

    def run(self, df):
        self.df = df

        def objective(position):
            sampling_strategy = float(position[0])
            try:
                return self._evaluate_params(df, sampling_strategy)
            except Exception as e:
                print(f"Erro ao avaliar: {e}")
                return 0.0

        bounds = [(0.1, 0.9)]  # intervalo válido para sampling_strategy
        optimizer = WhaleOptimizer(objective, bounds, n_whales=self.n_whales, n_iterations=self.n_iterations, seed=self.random_state)
        best_pos, best_score = optimizer.optimize()

        self.best_params = {"sampling_strategy": float(best_pos[0])}
        self.best_score = best_score

        print("\nBest parameters (WOA):")
        print(self.best_params)
        print(f"Mean AUC: {self.best_score:.4f}")

        return {
            'best_params': self.best_params,
            'best_cross_validation_score': self.best_score
        }
