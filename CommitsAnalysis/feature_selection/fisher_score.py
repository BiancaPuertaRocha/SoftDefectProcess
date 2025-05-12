import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FisherScoreFeatureSelector:
    """
    Feature selection using Fisher Score with hyperparameter tuning via Bayesian Optimization (Optuna).
    A customizable classifier is used for evaluation.
    """
    def __init__(self, classifier, n_trials=20, direction="maximize", sampler=None, min_features=3):
        self.classifier = classifier
        self.n_trials = n_trials
        self.direction = direction
        self.sampler = sampler
        self.study = None
        self.best_k = None
        self.best_score = None
        self.selected_features = None
        self.min_features = min_features

    def _fisher_score(self, X, y):
        """
        Compute Fisher score for each feature in X.
        """
        n_features = X.shape[1]
        fisher_scores = []

        for i in range(n_features):
            feature = X.iloc[:, i]
            mean_class_0 = feature[y == 0].mean()
            mean_class_1 = feature[y == 1].mean()
            var_class_0 = feature[y == 0].var()
            var_class_1 = feature[y == 1].var()

            fisher_score = (mean_class_0 - mean_class_1) ** 2 / (var_class_0 + var_class_1)
            fisher_scores.append(fisher_score)

        return fisher_scores

    def _evaluate_k(self, df, k):
        X = df.drop(columns=['failure_prone'])
        y = df['failure_prone']

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute Fisher Scores for all features
        # getting back the indexes to compare. After fit_transform the X loses the index
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        fisher_scores = self._fisher_score(X_scaled_df, y)

        # Select the top k features based on Fisher score
        top_k_features = np.argsort(fisher_scores)[-k:]
        selected_features = X.columns[top_k_features]

        # Cross-validation with the provided classifier
        X_selected = X[selected_features]
        score = cross_val_score(self.classifier, X_selected, y, cv=5, scoring='accuracy').mean()

        return score, selected_features

    def _objective(self, trial):
        X = self.df.drop(columns=['failure_prone'])
        max_k = X.shape[1]
        k = trial.suggest_int("k", self.min_features, max_k)

        try:
            score, selected_features = self._evaluate_k(self.df, k)
        except Exception as e:
            print(e)
            return 0.0
        return score

    def run(self, df):
        self.df = df
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials)

        # Best k and the corresponding score
        self.best_k = self.study.best_params["k"]
        self.best_score = self.study.best_value
        
        # Run the evaluation again to get the best features
        _, selected_features = self._evaluate_k(self.df, self.best_k)
        self.selected_features = selected_features

        print(f"\nBest number of features (k): {self.best_k}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        print("Selected features:")
        print(self.selected_features)

        return {
            'features': self.selected_features,
            'best_k': self.best_k,
            'best_cross_validation_score': self.best_score
        }
