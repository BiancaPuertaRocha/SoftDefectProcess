import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.model_selection import train_test_split

from ..data_balance.adasyn import ADASYNBalancer
from ..data_balance.random_undersampling import RandomUnderSamplerBalancer
from ..data_balance.smotee import SmoteeFeatureBalancer

from ..feature_selection.chi_square import Chi2FeatureSelector
from ..feature_selection.fisher_score import FisherScoreFeatureSelector
from ..feature_selection.ga import GAFeatureSelector

from preprocessor import Preprocessor

class MainPreprocessorRunner:
    def __init__(self, df, model=None, test_size=0.2, random_state=42):
        self.df = df
        self.model = model or RandomForestClassifier(random_state=random_state)
        self.test_size = test_size
        self.random_state = random_state

        # Estratégias de seleção de atributos
        self.fs_strategies = [
            FisherScoreFeatureSelector(self.model),
            Chi2FeatureSelector(self.model),
            GAFeatureSelector(self.model)
        ]

        # Estratégias de balanceamento
        self.balancer_strategies = [
            ADASYNBalancer(self.model),
            RandomUnderSamplerBalancer(self.model),
            SmoteeFeatureBalancer(self.model)
        ]

    def run_all(self):
        resultados = []
        results_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "results.csv")

        for fs in self.fs_strategies:
            for balancer in self.balancer_strategies:
                print("=" * 60)
                print(f">>> FS: {fs.__class__.__name__} + Balancer: {balancer.__class__.__name__}")

                preprocessor = Preprocessor(fs_strategy=fs, balancer_strategy=balancer)

                try:
                    X_resampled, y_resampled, info = preprocessor.run(self.df)

                    # Dividir em treino e teste
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_resampled, y_resampled, test_size=self.test_size, random_state=self.random_state
                    )

                    # Treinar modelo
                    self.model.fit(X_train, y_train)

                    # Prever
                    y_pred = self.model.predict(X_test)
                    y_proba = self.model.predict_proba(X_test)[:, 1]  # Para AUC

                    # Avaliar
                    auc = roc_auc_score(y_test, y_proba)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

                    # Obter métricas por classe
                    class_report = classification_report(
                        y_test, y_pred, output_dict=True, zero_division=0
                    )

                    # Exemplo: class_report['0'] contém métricas da classe 0
                    #          class_report['1'] contém métricas da classe 1
                    row = {
                        "fs": fs.__class__.__name__,
                        "balancer": balancer.__class__.__name__,
                        "selected_features": info["selected_features"],
                        "best_params": info["best_params"],
                        "auc": auc,
                        "accuracy": acc,
                        "precision": prec,
                        "recall": recall,
                        "f1": f1,
                        "class_0": {
                            "precision": class_report["0"]["precision"],
                            "recall": class_report["0"]["recall"],
                            "f1": class_report["0"]["f1-score"]
                        },
                        "class_1": {
                            "precision": class_report["1"]["precision"],
                            "recall": class_report["1"]["recall"],
                            "f1": class_report["1"]["f1-score"]
                        }
                    }
                    resultados.append(row)
                    df_row = pd.DataFrame([row])
                    df_row.to_csv(results_path, mode='a', index=False, header=False)

                except Exception as e:
                    print(f"Erro com combinação {fs.__class__.__name__} + {balancer.__class__.__name__}: {e}")

        return resultados
