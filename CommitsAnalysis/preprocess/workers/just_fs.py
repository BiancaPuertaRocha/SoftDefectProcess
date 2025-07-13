import pandas as pd
import os
import argparse

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from preprocess.utils.fs_runner import FSRunner

MIN_FEATURES = 7
TEXT_COLUMNS = ["message", "bug_message", "code_smell_message"]
EVAL_METHOD = 'f1'


def save_eval_result(filename, score, algorithm, output_dir="data"):
    result_df = pd.DataFrame([{
        "filename": filename,
        "eval_method_score": score,
        "algorithm": algorithm
    }])
    output_path = os.path.join(output_dir, "model_eval_results.csv")

    if os.path.exists(output_path):
        result_df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        result_df.to_csv(output_path, index=False)

# -------- Classifier Factories --------

def create_rf(): return RandomForestClassifier(n_estimators=50, random_state=42)

def create_bagging(base): return BaggingClassifier(estimator=base, n_estimators=10, random_state=42)

def create_cart(): return DecisionTreeClassifier(random_state=42)

def create_voting():
    return VotingClassifier(estimators=[
        ('cart', create_cart()),
        ('knn', KNeighborsClassifier()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('nb', GaussianNB()),
        ('rf', create_rf()),
        ('svm', SVC(probability=True, random_state=42))
    ], voting='soft')

# -------- Execution Logic --------

def run_with_selector(df, csv_filename, clf, model_name, selector: str):
    runner = FSRunner(clf=clf, model_name=model_name, min_features=MIN_FEATURES, eval_method=EVAL_METHOD)
    print(df.nunique())
    getattr(runner, f"run_{selector}")(df, csv_filename)

def run_all_methods(df: pd.DataFrame, csv_filename: str):
    selectors = ['fisher', 'chi',  'ga']
    configs = [
        (create_rf, 'random_forest'),
        (lambda: create_bagging(create_rf()), 'bagging_random_forest'),
        (lambda: create_bagging(create_cart()), 'bagging_dt'),
        (create_voting, 'voting'),
    ]

    for selector in selectors:
        for clf_func, model_name in configs:
            print(f"Running {selector.upper()} + {model_name}")
            run_with_selector(df, csv_filename, clf_func(), model_name, selector)

def _get_directories():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "data", "logs")
    fs_dir = os.path.join(base_dir, "data", "datasets")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(fs_dir, exist_ok=True)
    return logs_dir, fs_dir

def run_no_preprocess(df: pd.DataFrame, csv_filename: str):
    X = df.drop(columns=['failure_prone'])
    y = df['failure_prone']

    logs_dir, _ = _get_directories()

    models = [
        (create_rf(), 'random_forest'),
        (create_bagging(create_rf()), 'bagging_random_forest'),
        (create_bagging(create_cart()), 'bagging_dt'),
        (create_voting(), 'voting')
    ]

    all_metrics = []

    for model, name in models:
        print(f"Running without FS: {name}")

        score = cross_val_score(model, X, y, cv=5, scoring=EVAL_METHOD).mean()
        y_pred = cross_val_predict(model, X, y, cv=5)
        y_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)

        auc_1 = roc_auc_score(y, y_proba)
        auc_0 = roc_auc_score(1 - y, 1 - y_proba)

        metrics = {
            'filename': csv_filename,
            'eval_method_score': score,
            'algorithm': name,

            # Classe 0
            'precision_0': report['0']['precision'],
            'recall_0': report['0']['recall'],
            'f1_0': report['0']['f1-score'],
            'auc_0': auc_0,

            # Classe 1
            'precision_1': report['1']['precision'],
            'recall_1': report['1']['recall'],
            'f1_1': report['1']['f1-score'],
            'auc_1': auc_1,

            # Métricas gerais 
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],

            # Accuracy geral
            'accuracy': report['accuracy']
        }

        all_metrics.append(metrics)

    # Salva as métricas em um CSV
    df_metrics = pd.DataFrame(all_metrics)
    output_path = os.path.join(logs_dir, f"no_preprocessing_{csv_filename}")
    df_metrics.to_csv(output_path, index=False)
    print(f"\nResultados salvos em: {output_path}")
    

# -------- Main --------

def main():
    parser = argparse.ArgumentParser(description="Run feature selection + ensemble.")
    parser.add_argument("csv_filename", help="Path to the CSV dataset")
    parser.add_argument("function_name", help="Function name to run (e.g., run_ga_random_forest, run_all, run_no_preprocess)")

    parser.add_argument("--run_count", action="store_true", help="Count and print the number of examples per class")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    print("Skipping dataset preparation. Reading CSV as-is.")
    df = pd.read_csv(args.csv_filename)
    if args.run_count:
        if 'failure_prone' in df.columns:
            counts = df['failure_prone'].value_counts().sort_index()
            print("Class counts for 'failure_prone':")
            for cls, count in counts.items():
                print(f"  Class {cls}: {count}")
        else:
            print("Column 'failure_prone' not found in dataframe.")

    if args.function_name == "run_all":
        run_all_methods(df, args.csv_filename)
    elif args.function_name == "run_no_preprocess":
        run_no_preprocess(df, args.csv_filename)
    elif args.function_name in globals():
        print(f"Running function '{args.function_name}'")
        globals()[args.function_name](df, args.csv_filename)
    else:
        print(f"Function '{args.function_name}' not found.")

if __name__ == "__main__":
    main()
