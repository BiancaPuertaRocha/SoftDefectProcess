import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from fs_runner import FSRunner

MIN_FEATURES = 7
TEXT_COLUMNS = ["message", "bug_message", "code_smell_message"]


# -------- Utils --------


def remove_columns_with_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.nunique(dropna=False) > 1]

def apply_tfidf_to_messages(df: pd.DataFrame, max_features: int = 20) -> pd.DataFrame:
    """Apply TF-IDF to text columns and return DataFrame with enriched features."""
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

        # Isola target se existir
        target = chunk[['failure_prone']] if 'failure_prone' in chunk.columns else pd.DataFrame()

        numeric_df = chunk.select_dtypes(include='number')
        text_df = chunk[[col for col in TEXT_COLUMNS if col in chunk.columns]]
        combined = pd.concat([numeric_df, text_df], axis=1)

        if not target.empty:
            combined = pd.concat([combined, target], axis=1)

        all_chunks.append(combined)

    df = pd.concat(all_chunks, ignore_index=True)

    # Aplica TF-IDF
    df, tfidf_columns = apply_tfidf_to_messages(df)

    # Salva a target e remove antes de tratar os NaNs
    if 'failure_prone' in df.columns:
        target_col = df['failure_prone']
        df = df.drop(columns=['failure_prone'])
    else:
        target_col = pd.Series(index=df.index, data=None, name='failure_prone')

    # Remove linhas com NaN que não estejam nas colunas de TF-IDF
    non_tfidf_columns = [col for col in df.columns if col not in tfidf_columns]
    df = df.dropna(subset=non_tfidf_columns)

    # Reanexa a target (sem alterar seu conteúdo original)
    target_col = target_col.loc[df.index]
    df = pd.concat([df, target_col], axis=1)

    # Remove colunas com valores únicos (incluindo NaN)
    df = remove_columns_with_unique_values(df)

    return df

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
    runner = FSRunner(clf=clf, model_name=model_name, min_features=MIN_FEATURES)
    getattr(runner, f"run_{selector}")(df, csv_filename)


def run_all_methods(df: pd.DataFrame, csv_filename: str):
    selectors = [ 'fisher', 'chi', 'ga']
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


# -------- Main --------

def main():
    parser = argparse.ArgumentParser(description="Run feature selection + ensemble.")
    parser.add_argument("csv_filename", help="Path to the CSV dataset")
    parser.add_argument("function_name", help="Function name to run (e.g., run_ga_random_forest, run_all)")
    args = parser.parse_args()

    df = prepare_dataframe(args.csv_filename)
    print(f"Processed DataFrame: {len(df)} rows.")

    if args.function_name == "run_all":
        run_all_methods(df, args.csv_filename)
    elif args.function_name in globals():
        print(f"Running function '{args.function_name}'")
        globals()[args.function_name](df, args.csv_filename)
    else:
        print(f"Function '{args.function_name}' not found.")


if __name__ == "__main__":
    main()
