import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from data_balance.whale_optimizer import WhaleOptimizer

class SmoteeFeatureBalancer:
    """
    Applies SMOTEE with hyperparameter optimization using Whale Optimization Algorithm.
    Optimizes sampling_strategy and k_neighbors for SMOTE.
    """
    def __init__(self, classifier, n_whales=10, n_iterations=20, direction="maximize", random_state=42):
        self.classifier = classifier
        self.n_whales = n_whales
        self.n_iterations = n_iterations
        self.direction = direction
        self.random_state = random_state
        self.best_params = None
        self.best_score = None

    def _evaluate_params(self, df, sampling_strategy, k_neighbors, return_data=False):
        df_clean = df.dropna()
        X = df_clean.drop(columns=['failure_prone'])
        y = df_clean['failure_prone']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        smote = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
        smote_enn = SMOTEENN(
            smote=smote,
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
        )
        X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)

        if return_data:
            return X_resampled, y_resampled

        score = cross_val_score(self.classifier, X_resampled, y_resampled, cv=5, scoring='f1').mean()
        return score

    def run(self, df):
        self.df = df

        def objective(position):
            # position[0] = sampling_strategy (float)
            # position[1] = k_neighbors (float, arredondar e limitar)
            sampling_strategy = float(position[0])
            k_neighbors = int(np.round(position[1]))
            k_neighbors = np.clip(k_neighbors, 2, 10)
            try:
                score = self._evaluate_params(self.df, sampling_strategy, k_neighbors)
            except Exception as e:
                print("Erro durante avaliação:", e)
                score = 0.0
            return score

        bounds = [(0.1, 1.0), (2, 10)] 

        optimizer = WhaleOptimizer(objective, bounds, n_whales=self.n_whales, n_iterations=self.n_iterations, seed=self.random_state)
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
