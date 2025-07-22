import os, csv, datetime, uuid
import time
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

from preprocess.models.adasyn import ADASYNBalancer
from preprocess.models.random_undersampling import RandomUnderSamplerBalancer
from preprocess.models.smotee import SmoteeFeatureBalancer

from preprocess.models.chi_square import Chi2FeatureSelector
from preprocess.models.fisher_score import FisherScoreFeatureSelector
from preprocess.models.ga import GAFeatureSelector

from preprocess.utils.fs_db_runner import Preprocessor


class MainPreprocessorRunner:
    """
    Executa combinações de seleção de atributos + balanceamento,
    treina o modelo e registra métricas e tempos (em ms com precisão de µs).
    """
    def __init__(self, df, model, test_size=0.2, random_state=42, filename=''):
        self.df           = df
        self.filename     = filename
        self.model        = clone(model)
        self.test_size    = test_size
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
        
    # ------------------------------------------------------------------ #
    def _ns_to_ms(self, ns: int) -> float:
        """Converte nanossegundos em milissegundos (mantém as casas decimais)."""
        return ns / 1_000_000.0

    def _append_csv(self, row: dict):
        """Cria o CSV se não existir e adiciona `row`."""
        results_dir = '/home/bianca/SoftDefectProcess/CommitsAnalysis/data/logs'
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "time.csv")
        file_exists = os.path.isfile(result_file)

        # newline='' evita linhas em branco extras no Windows
        with open(result_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "id", "time_to_execute"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


    # ------------------------------------------------------------------ #
    def run_all(self):
        start = time.time()
        resultados = []
        results_dir = '/home/bianca/SoftDefectProcess/CommitsAnalysis/data/logs'
        os.makedirs(results_dir, exist_ok=True)

        id_execucao = str(uuid.uuid4())
        result_filename = f"{self.filename}__{self.model.__class__.__name__}__{id_execucao}__results.csv"
        results_path = os.path.join(results_dir, result_filename)

        print("Salvando resultado em: " + results_path)

        for fs in self.fs_strategies:
            for balancer in self.balancer_strategies:
                print("=" * 60)
                print(f">>> FS: {fs.__class__.__name__} + Balancer: {balancer.__class__.__name__}")

                # Pré‑processamento (seleção de atributos + balanceamento)
                t0_ns = time.perf_counter_ns()
                preprocessor = Preprocessor(fs_strategy=fs, balancer_strategy=balancer)
                X_resampled, y_resampled, info = preprocessor.run(self.df)
                tunning_and_preprocess_time_ms = self._ns_to_ms(time.perf_counter_ns() - t0_ns)

                # ----------------------------- treino -----------------------------
                t0_ns = time.perf_counter_ns()
                X_train, X_test, y_train, y_test = train_test_split(
                    X_resampled, y_resampled,
                    test_size=self.test_size,
                    random_state=self.random_state
                )

                model = clone(self.model)
                model.fit(X_train, y_train)
                train_time_ms = self._ns_to_ms(time.perf_counter_ns() - t0_ns)

                # --------------------------- predição -----------------------------
                t0_ns = time.perf_counter_ns()
                y_pred  = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                predict_time_ms = self._ns_to_ms(time.perf_counter_ns() - t0_ns)

                # ---------------------------- métricas ----------------------------
                auc     = roc_auc_score(y_test, y_proba)
                acc     = accuracy_score(y_test, y_pred)
                prec    = precision_score(y_test, y_pred, zero_division=0)
                recall  = recall_score(y_test, y_pred, zero_division=0)
                f1      = f1_score(y_test, y_pred, zero_division=0)

                class_report = classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0
                )

                # ---------------------------- registro ----------------------------
                row = {
                    "fs": fs.__class__.__name__,
                    "balancer": balancer.__class__.__name__,
                    "selected_features": info["features"],
                    "best_params": info["best_params"],

                    "auc": auc,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": recall,
                    "f1": f1,

                    "precision_0": class_report["0"]["precision"],
                    "recall_0":    class_report["0"]["recall"],
                    "f1_0":        class_report["0"]["f1-score"],
                    "precision_1": class_report["1"]["precision"],
                    "recall_1":    class_report["1"]["recall"],
                    "f1_1":        class_report["1"]["f1-score"],

                    # tempos em milissegundos com precisão de microssegundos
                    "train_time_ms":   train_time_ms,
                    "predict_time_ms": predict_time_ms,
                    "tunning_and_preprocess_time_ms": tunning_and_preprocess_time_ms
                }

                resultados.append(row)

                # Grava incrementalmente garantindo 9 casas decimais
                df_row = pd.DataFrame([row])
                write_header = not os.path.exists(results_path)
                df_row.to_csv(
                    results_path,
                    mode="a",
                    index=False,
                    header=write_header,
                    float_format="%.9f"
                )
        elapsed = time.time() - start
        self._append_csv({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "id": result_filename,
            "time_to_execute": elapsed
        })
        return resultados
