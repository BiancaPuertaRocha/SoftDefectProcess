import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FisherScoreFeatureSelector:
    """
    Feature selection using Fisher Score combined with Bayesian Optimization (Optuna)
    to tune the optimal number of features (k) that maximize classifier performance.
    """

    def __init__(self, classifier, n_trials=20, direction="maximize", sampler=None, min_features=3, eval_method='roc_auc'):
        """
        Initialize the feature selector.

        Args:
            classifier: A scikit-learn compatible classifier to evaluate selected features.
            n_trials: Number of Optuna trials for hyperparameter search.
            direction: Direction for optimization ('maximize' or 'minimize').
            sampler: Optional Optuna sampler to control sampling strategy.
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
        self.min_features = min_features
        self.eval_method = eval_method

    def _fisher_score(self, X, y):
        """
        Compute the Fisher score for each feature.

        Args:
            X: Feature matrix as a pandas DataFrame.
            y: Target vector as a 1D array or pandas Series.

        Returns:
            List of Fisher scores, one per feature.
        """
        n_features = X.shape[1]
        fisher_scores = []

        for i in range(n_features):
            feature = X.iloc[:, i]
            # Calculate mean of feature values for each class
            mean_class_0 = feature[y == 0].mean()
            mean_class_1 = feature[y == 1].mean()
            # Calculate variance of feature values for each class
            var_class_0 = feature[y == 0].var()
            var_class_1 = feature[y == 1].var()

            # Calculate Fisher score for this feature
            fisher_score = (mean_class_0 - mean_class_1) ** 2 / (var_class_0 + var_class_1)
            fisher_scores.append(fisher_score)

        return fisher_scores

    def _evaluate_k(self, df, k):
        """
        Evaluate model performance using top-k features selected by Fisher score.

        Args:
            df: DataFrame containing features and target 'failure_prone'.
            k: Number of features to select.

        Returns:
            Tuple: (mean cross-validation ROC-AUC score, selected feature names)
        """
        # Separate features and target
        X = df.drop(columns=['failure_prone'])
        y = df['failure_prone']

        # Encode target if categorical
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Scale features with StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Convert scaled features back to DataFrame to keep column names
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Calculate Fisher scores for each feature
        fisher_scores = self._fisher_score(X_scaled_df, y)

        # Select indices of top k features with highest Fisher scores
        top_k_features = np.argsort(fisher_scores)[-k:]
        selected_features = X.columns[top_k_features]

        # Evaluate classifier performance with selected features via cross-validation
        X_selected = X[selected_features]
        score = cross_val_score(self.classifier, X_selected, y, cv=5, scoring=self.eval_method).mean()

        return score, selected_features

    def _objective(self, trial):
        """
        Optuna objective function to suggest k and evaluate performance.

        Args:
            trial: Optuna trial object.

        Returns:
            Cross-validation ROC-AUC score for the selected k.
        """
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
        """
        Run the Bayesian optimization to find the best k and select features.

        Args:
            df: DataFrame containing features and target column 'failure_prone'.

        Returns:
            Dictionary with keys 'features', 'best_k', and 'best_cross_validation_score'.
        """
        self.df = df

        # Create and run Optuna study to optimize k
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials)

        # Extract best number of features and score
        self.best_k = self.study.best_params["k"]
        self.best_score = self.study.best_value
        
        # Get selected features for best k
        _, selected_features = self._evaluate_k(self.df, self.best_k)
        self.selected_features = selected_features

        # Print summary of results
        print(f"\nBest number of features (k): {self.best_k}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        print("Selected features:")
        print(self.selected_features)

        return {
            'features': self.selected_features,
            'best_params': self.best_k,
            'best_score': self.best_score
        }
