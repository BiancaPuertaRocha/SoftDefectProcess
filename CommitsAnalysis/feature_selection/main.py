import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
import argparse
from fs_runner import FSRunner

MIN_FEATURES = 7
TEXT_COLUMNS = ["message", "bug_message", "code_smell_message"]
EVAL_METHOD = 'roc_auc'

# -------- Utils --------

def remove_columns_with_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.nunique(dropna=False) > 1]

# ignore if you used the 9_new_features.py
def apply_tfidf_to_messages(df: pd.DataFrame, max_features: int = 20) -> pd.DataFrame:
    for col in TEXT_COLUMNS:
        df[col] = df.get(col, "").fillna("")

    tfidf_frames = []
    tfidf_column_names = []

    for col in TEXT_COLUMNS:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(df[col])
        feature_names = [f"{col}_{word}" for word in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        tfidf_frames.append(tfidf_df)
        tfidf_column_names.extend(feature_names)

    df = df.reset_index(drop=True)
    tfidf_frames = [f.reset_index(drop=True) for f in tfidf_frames]

    df = pd.concat([df] + tfidf_frames, axis=1)
    df = df.drop(columns=TEXT_COLUMNS)

    return df, tfidf_column_names

def prepare_dataframe(csv_filename: str) -> pd.DataFrame:
    all_chunks = []

    for chunk in pd.read_csv(csv_filename, chunksize=10000):
        for label_col in ['bug_status', 'smell_status']:
            if label_col in chunk.columns:
                chunk[label_col] = chunk[label_col].astype('category').cat.codes

        numeric_df = chunk.select_dtypes(include='number')
        text_df = chunk[[col for col in TEXT_COLUMNS if col in chunk.columns]]
        cols_to_add = [col for col in text_df.columns if col not in numeric_df.columns]
        combined = pd.concat([numeric_df, text_df[cols_to_add]], axis=1)

        all_chunks.append(combined)

    df = pd.concat([chunk.reset_index(drop=True) for chunk in all_chunks], ignore_index=True)

    df, tfidf_columns = apply_tfidf_to_messages(df)

    if 'failure_prone' in df.columns:
        target_col = df['failure_prone']
        df = df.drop(columns=['failure_prone'])
    else:
        target_col = pd.Series(index=df.index, data=None, name='failure_prone')

    non_tfidf_columns = [col for col in df.columns if col not in tfidf_columns]
    
    # 1. Identify columns where more than half of the values are NaN
    cols_to_drop = [col for col in non_tfidf_columns if df[col].isna().mean() > 0.5]

    # 2. Drop these columns from the DataFrame
    df = df.drop(columns=cols_to_drop)

    # 3. Get the remaining non-TFIDF columns after dropping
    cols_to_check = [col for col in non_tfidf_columns if col not in cols_to_drop]

    # 4. Drop rows that have NaN values in any of the remaining columns
    df = df.dropna(subset=cols_to_check)

    target_col = target_col.loc[df.index]
    df = pd.concat([df, target_col], axis=1)

    df = remove_columns_with_unique_values(df)

    return df

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
        (create_voting, 'voting'),
        (create_rf, 'random_forest'),
        (lambda: create_bagging(create_rf()), 'bagging_random_forest'),
        (lambda: create_bagging(create_cart()), 'bagging_dt'),
    ]

    for selector in selectors:
        for clf_func, model_name in configs:
            print(f"Running {selector.upper()} + {model_name}")
            run_with_selector(df, csv_filename, clf_func(), model_name, selector)


def run_no_preprocess(df: pd.DataFrame, csv_filename: str):
    X = df.drop(columns=['failure_prone'])
    y = df['failure_prone']

    os.makedirs("data", exist_ok=True)

    models = [
        (create_rf(), 'random_forest'),
        (create_bagging(create_rf()), 'bagging_random_forest'),
        (create_bagging(create_cart()), 'bagging_dt'),
        (create_voting(), 'voting')
    ]

    results = []

    for model, name in models:
        print(f"Running without FS: {name}")

        # Score global com a métrica principal
        score = cross_val_score(model, X, y, cv=5, scoring=EVAL_METHOD).mean()

        # Predições e probabilidades para avaliação detalhada
        y_pred = cross_val_predict(model, X, y, cv=5)
        y_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]

        # Relatório detalhado com output_dict para fácil extração
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)

        # Calcula AUC para ambas as classes
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

            # Métricas gerais (macro average)
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
        }

        results.append(metrics)

    results_df = pd.DataFrame(results)

    if os.path.exists(csv_filename):
        results_df.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        results_df.to_csv(csv_filename, index=False)

# -------- Main --------

def main():
    parser = argparse.ArgumentParser(description="Run feature selection + ensemble.")
    parser.add_argument("csv_filename", help="Path to the CSV dataset")
    parser.add_argument("function_name", help="Function name to run (e.g., run_ga_random_forest, run_all, run_no_preprocess)")
    parser.add_argument("--skip_prepare", action="store_true", help="Skip the preprocessing step and use the raw DataFrame as-is")
    parser.add_argument("--run_count", action="store_true", help="Count and print the number of examples per class")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    if args.skip_prepare:
        print("Skipping dataset preparation. Reading CSV as-is.")
        df = pd.read_csv(args.csv_filename)
    else:
        df = prepare_dataframe(args.csv_filename)
        print(f"Processed DataFrame: {len(df)} rows.")

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
