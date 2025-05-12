import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import argparse
from fs_runner import FSRunner


MIN_FEATURES = 5

def remove_columns_with_unique_values(df):
    """Remove columns that have only one unique value."""
    return df.loc[:, df.nunique() > 1]


# RANDOM FOREST

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


# BAGGING + RANDOM FOREST

def run_chi_bagging_random_forest(df, csv_filename):
    base_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
    runner = FSRunner(clf=clf, model_name='bagging_random_forest', min_features=MIN_FEATURES)
    runner.run_chi(df, csv_filename)

def run_fisher_bagging_random_forest(df, csv_filename):
    base_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
    runner = FSRunner(clf=clf, model_name='bagging_random_forest', min_features=MIN_FEATURES)
    runner.run_fisher(df, csv_filename)

def run_ga_bagging_random_forest(df, csv_filename):
    base_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
    runner = FSRunner(clf=clf, model_name='bagging_random_forest', min_features=MIN_FEATURES)
    runner.run_ga(df, csv_filename)


# BAGGING + DECISION TREE (CART)

def run_chi_bagging_cart(df, csv_filename):
    base_clf = DecisionTreeClassifier(random_state=42)
    clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
    runner = FSRunner(clf=clf, model_name='bagging_dt', min_features=MIN_FEATURES)
    runner.run_chi(df, csv_filename)

def run_fisher_bagging_cart(df, csv_filename):
    base_clf = DecisionTreeClassifier(random_state=42)
    clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
    runner = FSRunner(clf=clf, model_name='bagging_dt', min_features=MIN_FEATURES)
    runner.run_fisher(df, csv_filename)

def run_ga_bagging_cart(df, csv_filename):
    base_clf = DecisionTreeClassifier(random_state=42)
    clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
    runner = FSRunner(clf=clf, model_name='bagging_dt', min_features=MIN_FEATURES)
    runner.run_ga(df, csv_filename)


# VOTING ENSEMBLE

def create_voting_classifier():
    """Create a soft-voting ensemble with diverse base classifiers."""
    estimators = [
        ('cart', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('nb', GaussianNB()),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ]
    return VotingClassifier(estimators=estimators, voting='soft')

def run_chi_voting(df, csv_filename):
    clf = create_voting_classifier()
    runner = FSRunner(clf=clf, model_name='voting', min_features=MIN_FEATURES)
    runner.run_chi(df, csv_filename)

def run_fisher_voting(df, csv_filename):
    clf = create_voting_classifier()
    runner = FSRunner(clf=clf, model_name='voting', min_features=MIN_FEATURES)
    runner.run_fisher(df, csv_filename)

def run_ga_voting(df, csv_filename):
    clf = create_voting_classifier()
    runner = FSRunner(clf=clf, model_name='voting', min_features=MIN_FEATURES)
    runner.run_ga(df, csv_filename)


# TF-IDF TEXT TRANSFORMATION

def apply_tfidf_to_messages(df, max_features=20):
    """Apply TF-IDF transformation to text columns and return enriched DataFrame."""
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

    df = df.reset_index(drop=True)
    df_bug = df_bug.reset_index(drop=True)
    df_msg = df_msg.reset_index(drop=True)
    df_smell = df_smell.reset_index(drop=True)

    return pd.concat([df, df_bug, df_msg, df_smell], axis=1)


# MAIN EXECUTION FUNCTION

def run_all_methods(df, csv_filename):
    """Run all combinations of feature selection and ensemble methods."""
    print("Running ALL methods sequentially...\n")

    # Random Forest
    run_fisher_random_forest(df, csv_filename)
    run_chi_random_forest(df, csv_filename)
    run_ga_random_forest(df, csv_filename)

    # Bagging + Random Forest
    run_fisher_bagging_random_forest(df, csv_filename)
    run_chi_bagging_random_forest(df, csv_filename)
    run_ga_bagging_random_forest(df, csv_filename)

    # Bagging + Decision Tree (CART)
    run_fisher_bagging_cart(df, csv_filename)
    run_chi_bagging_cart(df, csv_filename)
    run_ga_bagging_cart(df, csv_filename)

    # Voting Ensemble
    run_fisher_voting(df, csv_filename)
    run_chi_voting(df, csv_filename)
    run_ga_voting(df, csv_filename)


def prepare_data_and_run_function(csv_filename, function_name):
    """
    Prepare dataset by transforming text fields with TF-IDF, removing low-variance columns,
    and executing the selected function or all.
    """
    all_chunks = []
    text_columns = ["message", "bug_message", "code_smell_message"]

    for chunk in pd.read_csv(csv_filename, chunksize=10000):
        if 'bug_status' in chunk.columns and 'smell_status' in chunk.columns:
            chunk['bug_status'] = chunk['bug_status'].astype('category').cat.codes
            chunk['smell_status'] = chunk['smell_status'].astype('category').cat.codes

        numeric_df = chunk.select_dtypes(include='number')
        existing_text_columns = [col for col in text_columns if col in chunk.columns]
        combined_chunk = pd.concat([numeric_df, chunk[existing_text_columns]], axis=1)
        all_chunks.append(combined_chunk)

    df = pd.concat(all_chunks, ignore_index=True)

    # Apply TF-IDF
    df = apply_tfidf_to_messages(df)
    df = remove_columns_with_unique_values(df)
    df = df.drop(columns=text_columns)

    print(f"\nProcessed DataFrame: {len(df)} rows.")

    # Execute the selected function
    if function_name == "run_all":
        run_all_methods(df, csv_filename)
    elif function_name in globals():
        print(f"Running function '{function_name}'")
        globals()[function_name](df, csv_filename)
    else:
        print(f"Function '{function_name}' not found.")


def main():
    parser = argparse.ArgumentParser(description="Run feature selection method on a dataset.")
    parser.add_argument("csv_filename", help="Path to the CSV dataset")
    parser.add_argument("function_name", help="Function name to run (e.g., run_ga_random_forest, run_all)")

    args = parser.parse_args()
    prepare_data_and_run_function(args.csv_filename, args.function_name)


if __name__ == "__main__":
    main()
