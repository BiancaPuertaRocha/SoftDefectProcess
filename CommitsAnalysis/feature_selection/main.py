import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import argparse

from fs_runner import FSRunner

def remove_columns_with_unique_values(df):
    """Removes columns that contain only a single unique value."""
    return df.loc[:, df.nunique() > 1]


#RANDOM FOREST

def run_fisher_random_forest(df, csv_filename):
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    runner = FSRunner(clf, 'random_forest')
    runner.run_fisher(df, csv_filename)

def run_chi_random_forest(df, csv_filename):
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    runner = FSRunner(clf, 'random_forest')
    runner.run_chi(df, csv_filename)

def run_ga_random_forest(df, csv_filename):
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    runner = FSRunner(clf, 'random_forest')
    runner.run_ga(df, csv_filename)


#BAGGING + RANDOM FOREST

def run_chi_bagging_random_forest(df, csv_filename):
    """
    Runs feature selection Chi Square using Bagging with Random Forest as base estimator.
    """
    base_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
    runner = FSRunner(clf, 'bagging_random_forest')
    runner.run_chi(df, csv_filename)

def run_fisher_bagging_random_forest(df, csv_filename):
    """
    Runs feature selection Fisher Score using Bagging with Random Forest as base estimator.
    """
    base_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
    runner = FSRunner(clf, 'bagging_random_forest')
    runner.run_fisher(df, csv_filename)


def run_ga_bagging_random_forest(df, csv_filename):
    """
    Runs feature selection Genetic Algorithm using Bagging with Random Forest as base estimator.
    """
    base_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
    runner = FSRunner(clf, 'bagging_random_forest')
    runner.run_ga(df, csv_filename)

def apply_tfidf_to_messages(df, max_features=20):
    """
    Applies TF-IDF to text columns: 'message', 'bug_message', and 'code_smell_message'.
    """
    df['bug_message'] = df['bug_message'].fillna('')
    df['message'] = df['message'].fillna('')
    df['code_smell_message'] = df['code_smell_message'].fillna('')

    tfidf_bug = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_msg = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_smell = TfidfVectorizer(stop_words='english', max_features=max_features)

    X_bug = tfidf_bug.fit_transform(df['bug_message']).toarray()
    X_msg = tfidf_msg.fit_transform(df['message']).toarray()
    X_smell = tfidf_smell.fit_transform(df['code_smell_message']).toarray()

    bug_vocab = tfidf_bug.get_feature_names_out()
    msg_vocab = tfidf_msg.get_feature_names_out()
    smell_vocab = tfidf_smell.get_feature_names_out()

    df_bug = pd.DataFrame(X_bug, columns=[f'bug_message_{word}' for word in bug_vocab])
    df_msg = pd.DataFrame(X_msg, columns=[f'message_{word}' for word in msg_vocab])
    df_smell = pd.DataFrame(X_smell, columns=[f'code_smell_message_{word}' for word in smell_vocab])

    # Reset index to safely concatenate
    df = df.reset_index(drop=True)
    df_bug = df_bug.reset_index(drop=True)
    df_msg = df_msg.reset_index(drop=True)
    df_smell = df_smell.reset_index(drop=True)

    return pd.concat([df, df_bug, df_msg, df_smell], axis=1)

def prepare_data_and_run_function(csv_filename, function_name):
    """
    Prepares the data by handling numeric fields and TF-IDF processing, 
    then runs the specified feature selection function.
    """
    all_chunks = []
    text_columns = ["message", "bug_message", "code_smell_message"]

    for chunk in pd.read_csv(csv_filename, chunksize=10000):
        # Convert categorical to numeric
        if 'bug_status' in chunk.columns and 'smell_status' in chunk.columns:
            chunk['bug_status'] = chunk['bug_status'].astype('category').cat.codes
            chunk['smell_status'] = chunk['smell_status'].astype('category').cat.codes

        numeric_df = chunk.select_dtypes(include='number')
        existing_text_columns = [col for col in text_columns if col in chunk.columns]
        combined_chunk = pd.concat([numeric_df, chunk[existing_text_columns]], axis=1)
        all_chunks.append(combined_chunk)

    df = pd.concat(all_chunks, ignore_index=True)

    # Apply TF-IDF to text
    df = apply_tfidf_to_messages(df)
    df = remove_columns_with_unique_values(df)
    df = df.drop(columns=text_columns)

    print(f"\nProcessed DataFrame: {len(df)} rows.")

    # Call the selected function
    if function_name in globals():
        print(f"Running function '{function_name}'")
        globals()[function_name](df, csv_filename)
    else:
        print(f"Function '{function_name}' not found.")

def main():
    parser = argparse.ArgumentParser(description="Run feature selection method on a dataset.")
    parser.add_argument("csv_filename", help="Path to the CSV dataset")
    parser.add_argument("function_name", help="Function name to run (e.g., run_ga_random_forest, run_chi_random_forest, run_bagging_random_forest)")

    args = parser.parse_args()
    prepare_data_and_run_function(args.csv_filename, args.function_name)

if __name__ == "__main__":
    main()
