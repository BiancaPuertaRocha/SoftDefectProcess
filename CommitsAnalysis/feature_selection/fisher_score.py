class FisherScoreFeatureSelector:
    def __init__(self, classifier, n_trials=20, direction="maximize", sampler=None):
        self.classifier = classifier
        self.n_trials = n_trials
        self.direction = direction
        self.sampler = sampler
        self.study = None
        self.best_k = None
        self.best_score = None
        self.selected_features = None

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

            # Fisher score calculation (between-class variance / within-class variance)
            fisher_score = (mean_class_0 - mean_class_1) ** 2 / (var_class_0 + var_class_1)
            fisher_scores.append(fisher_score)

        return fisher_scores

    def _evaluate_k(self, df, k):
        X = df.drop(columns=['failure_prone'])
        y = df['failure_prone']

        # Encode target if needed
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Fisher score requires non-negative values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute Fisher Scores for all features
        fisher_scores = self._fisher_score(pd.DataFrame(X_scaled, columns=X.columns), y)

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
        k = trial.suggest_int("k", 1, max_k)

        try:
            score, selected_features = self._evaluate_k(self.df, k)
        except Exception e:
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

        return self.selected_features
