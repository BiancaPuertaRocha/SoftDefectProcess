import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)


class BaggingDecisionTreeModel:
    def __init__(self, csv_filename, test_size=0.2, random_state=42):
        self.filename = csv_filename
        self.data = pd.read_csv(csv_filename)
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.model_name = "bagging_cart"

    def preprocess_data(self):
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def build_model(self):
        base_tree = DecisionTreeClassifier(random_state=self.random_state)
        self.model = BaggingClassifier(
            base_estimator=base_tree,
            n_estimators=10,
            random_state=self.random_state,
            n_jobs=-1
        )

    def train(self):
        self.preprocess_data()
        self.build_model()
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1] if hasattr(self.model, "predict_proba") else None

        metrics = {
            'dataset': os.path.basename(self.filename),
            'model': self.model_name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='binary'),
            'recall': recall_score(self.y_test, y_pred, average='binary'),
            'f1_score': f1_score(self.y_test, y_pred, average='binary'),
            'auc': None
        }

        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(self.y_test, y_proba)
            except ValueError:
                metrics['auc'] = None

        data_dir = os.path.dirname(os.path.abspath(self.filename))
        output_file = os.path.join(data_dir, 'metrics.csv')
        metrics_df = pd.DataFrame([metrics])

        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            if not ((existing['dataset'] == metrics['dataset']) &
                    (existing['modelo'] == metrics['modelo'])).any():
                metrics_df.to_csv(output_file, mode='a', index=False, header=False)
        else:
            metrics_df.to_csv(output_file, index=False)

        print(f"Evaluation saved in: {output_file}")
        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if v is not None and isinstance(v, float) else f"{k}: {v}")
