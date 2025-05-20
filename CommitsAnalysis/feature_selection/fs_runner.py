import os
import json
from datetime import datetime
from ga import GAFeatureSelector
from fisher_score import FisherScoreFeatureSelector
from chi_square import Chi2FeatureSelector

import pandas as pd

class FSRunner:

    def __init__(self, clf, model_name, min_features=3, eval_method='roc_auc'):
        self.clf = clf
        self.min_features = min_features
        self.model_name = model_name
        self.eval_method = eval_method

    def _remove_csv_extension(self, filename):
        return os.path.splitext(os.path.basename(filename))[0]

    def _get_directories(self, csv_filename):
        base_dir = os.path.dirname(csv_filename)
        logs_dir = os.path.join(base_dir, "logs")
        fs_dir = os.path.join(base_dir, "feature_selection")
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(fs_dir, exist_ok=True)
        return logs_dir, fs_dir

    def _save_data_log(self, data, method_name, base_filename, logs_dir):
        def convert(obj):
            if isinstance(obj, (set, tuple)):
                return list(obj)
            if isinstance(obj, pd.Index):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = os.path.join(logs_dir, f"{base_filename}_{method_name}_{self.model_name}_log_{now}.json")
        with open(log_filename, 'w') as f:
            json.dump(data, f, indent=4, default=convert)
        print(f"Log saved to: {log_filename}")


    def _save_selected_features(self, df, selected_features, method_name, base_filename, fs_dir):
        selected_df = df[selected_features]
        filename = os.path.join(fs_dir, f"{base_filename}_{method_name}_{self.model_name}_selected_features.csv")
        selected_df.to_csv(filename, index=False)
        print(f"Selected features saved to: {filename}")
        return selected_df

    def _remove_unwanted_columns(self, df):
        cols_to_remove = ['sha', 'filename', 'commit_sha', 'commit_date', 'branch_sonar']
        return df.drop(columns=[col for col in cols_to_remove if col in df.columns], errors='ignore')

    def _add_unwanted_columns_back(self, original_df, df):
        cols_to_add = ['sha', 'filename', 'commit_sha', 'commit_date', 'branch_sonar']
        for col in cols_to_add:
            if col in original_df.columns:
                df[col] = original_df[col]
        return df

    def run_fisher(self, df, csv_filename):
        method_name = "run_fisher"
        base_filename = self._remove_csv_extension(csv_filename)
        logs_dir, fs_dir = self._get_directories(csv_filename)

        # Salva as colunas originais para adicionar depois
        original_df = df.copy()

        # Remove as colunas que não interessam para seleção
        df = self._remove_unwanted_columns(df)

        selector = FisherScoreFeatureSelector(self.clf, min_features=self.min_features, eval_method=self.eval_method)
        data = selector.run(df)

        self._save_data_log(data, method_name, base_filename, logs_dir)
        selected_features = data['features']
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]

        # Adiciona as colunas removidas de volta antes de salvar
        selected_df = self._add_unwanted_columns_back(original_df, selected_df)

        filename = os.path.join(fs_dir, f"{base_filename}_{method_name}_{self.model_name}_selected_features.csv")
        selected_df.to_csv(filename, index=False)
        print(f"Selected features saved to: {filename}")
        return selected_features, selected_df

    def run_chi(self, df, csv_filename):
        method_name = "run_chi"
        base_filename = self._remove_csv_extension(csv_filename)
        logs_dir, fs_dir = self._get_directories(csv_filename)

        original_df = df.copy()
        df = self._remove_unwanted_columns(df)

        selector = Chi2FeatureSelector(self.clf, min_features=self.min_features, eval_method=self.eval_method)
        data = selector.run(df)

        self._save_data_log(data, method_name, base_filename, logs_dir)
        selected_features = data['features']
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]
        selected_df = self._add_unwanted_columns_back(original_df, selected_df)

        filename = os.path.join(fs_dir, f"{base_filename}_{method_name}_{self.model_name}_selected_features.csv")
        selected_df.to_csv(filename, index=False)
        print(f"Selected features saved to: {filename}")
        return selected_features, selected_df

    def run_ga(self, df, csv_filename):
        method_name = "run_ga"
        base_filename = self._remove_csv_extension(csv_filename)
        logs_dir, fs_dir = self._get_directories(csv_filename)

        original_df = df.copy()
        df = self._remove_unwanted_columns(df)

        selector = GAFeatureSelector(self.clf, min_features=self.min_features, eval_method=self.eval_method)
        data = selector.run(df)

        self._save_data_log(data, method_name, base_filename, logs_dir)
        selected_features = data['features']
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]
        selected_df = self._add_unwanted_columns_back(original_df, selected_df)

        filename = os.path.join(fs_dir, f"{base_filename}_{method_name}_{self.model_name}_selected_features.csv")
        selected_df.to_csv(filename, index=False)
        print(f"Selected features saved to: {filename}")
        return selected_features, selected_df