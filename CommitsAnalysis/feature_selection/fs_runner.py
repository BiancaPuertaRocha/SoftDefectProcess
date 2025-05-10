from ga import GAFeatureSelector
from fisher_score import FisherScoreFeatureSelector
from chi_square import Chi2FeatureSelector

class FSRunner:

    def __init__(self, clf, model_name, min_features=3):
        self.clf = clf
        self.min_features = min_features
        self.model_name = model_name

    def run_fisher(self, df, csv_filename):
        fisher_feature_selector = FisherScoreFeatureSelector(self.clf, min_features=self.min_features)
        selected_features = fisher_feature_selector.run(df)
        
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]

        function_name =f"run_fisher_{self.model_name}"
        filename = f"{csv_filename}_{function_name}_selected_features.csv"
        selected_df.to_csv(filename, index=False)

        print(f"Selected features saved to: {filename}")


    def run_chi(self, df, csv_filename):
        chi_feature_selector = Chi2FeatureSelector(self.clf, min_features=self.min_features)
        selected_features = chi_feature_selector.run(df)
        
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]

        function_name = f"run_chi_{self.model_name}"
        filename = f"{csv_filename}_{function_name}_selected_features.csv"
        selected_df.to_csv(filename, index=False)

        print(f"Selected features saved to: {filename}")


    def run_ga(self, df, csv_filename):
        ga_feature_selector = GAFeatureSelector(self.clf, min_features=self.min_features)
        selected_features = ga_feature_selector.run(df)
        
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]

        function_name = f"run_ga_{self.model_name}"
        filename = f"{csv_filename}_{function_name}_selected_features.csv"
        selected_df.to_csv(filename, index=False)

        print(f"Selected features saved to: {filename}")


    