from ga import GAFeatureSelector
from fisher_score import FisherScoreFeatureSelector
from chi_square import Chi2FeatureSelector

class FSRunner:

    def __init__(clf, model_name):
        self.clf = clf
        self.model_name = model_name

    def run_fisher(df, csv_filename):
        fisher_feature_selector = FisherScoreFeatureSelector(clf)
        selected_features = fisher_feature_selector.run(df)
        
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]

        function_name =f"run_fisher_{self.model_name}"
        filename = f"{csv_filename}_{function_name}_selected_features.csv"
        selected_df.to_csv(filename, index=False)

        print(f"Selected features saved to: {filename}")


    def run_chi(df, csv_filename):
        chi_feature_selector = Chi2FeatureSelector(clf)
        selected_features = chi_feature_selector.run(df)
        
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]

        function_name = f"run_chi_{self.model_name}"
        filename = f"{csv_filename}_{function_name}_selected_features.csv"
        selected_df.to_csv(filename, index=False)

        print(f"Selected features saved to: {filename}")


    def run_ga(df, csv_filename):
        ga_feature_selector = GAFeatureSelector(clf)
        selected_features = ga_feature_selector.run(df)
        
        print(f"Selected Features: {selected_features}")

        selected_df = df[selected_features]

        function_name = f"run_ga_{self.model_name}"
        filename = f"{csv_filename}_{function_name}_selected_features.csv"
        selected_df.to_csv(filename, index=False)

        print(f"Selected features saved to: {filename}")


    