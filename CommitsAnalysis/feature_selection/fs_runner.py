import os
import json
from datetime import datetime
from ga import GAFeatureSelector
from fisher_score import FisherScoreFeatureSelector
from chi_square import Chi2FeatureSelector

class FSRunner:

    def __init__(self, clf, model_name, min_features=3):
        self.clf = clf
        self.min_features = min_features
        self.model_name = model_name

    def _remove_csv_extension(self, filename):
        return os.path.splitext(filename)[0]

    def _save_data_log(self, data, method_name, base_filename):
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f"{base_filename}_{method_name}_{self.model_name}_log_{now}.json"
        with open(log_filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Log saved to: {log_filename}")

    def run_fisher(self, df, csv_filename):
        method_name = "run_fisher"
        base_filename = self._remove_csv_extension(csv_filename)
        fisher_feature_selector = FisherScoreFeatureSelector(self.clf, min_features=self.min_features)
        data = fisher_feature_selector.run(df)

        self._save_data_log(data, method_name, base_filename)

        selected_features = data['features']
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]
        filename = f"{base_filename}_{method_name}_{self.model_name}_selected_features.csv"
        selected_df.to_csv(filename, index=False)
        print(f"Selected features saved to: {filename}")

        return selected_features, selected_df

    def run_chi(self, df, csv_filename):
        method_name = "run_chi"
        base_filename = self._remove_csv_extension(csv_filename)
        chi_feature_selector = Chi2FeatureSelector(self.clf, min_features=self.min_features)
        data = chi_feature_selector.run(df)

        self._save_data_log(data, method_name, base_filename)

        selected_features = data['features']
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]
        filename = f"{base_filename}_{method_name}_{self.model_name}_selected_features.csv"
        selected_df.to_csv(filename, index=False)
        print(f"Selected features saved to: {filename}")

        return selected_features, selected_df

    def run_ga(self, df, csv_filename):
        method_name = "run_ga"
        base_filename = self._remove_csv_extension(csv_filename)
        ga_feature_selector = GAFeatureSelector(self.clf, min_features=self.min_features)
        data = ga_feature_selector.run(df)

        self._save_data_log(data, method_name, base_filename)

        selected_features = data['features']
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]
        filename = f"{base_filename}_{method_name}_{self.model_name}_selected_features.csv"
        selected_df.to_csv(filename, index=False)
        print(f"Selected features saved to: {filename}")

        return selected_features, selected_df
