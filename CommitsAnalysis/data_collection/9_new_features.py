"""
This script processes a CSV file containing commit-related data (bugs, code smells, patches, metrics, etc.).
Steps performed:
1. Removes irrelevant columns.
2. Propagates metric values across rows by 'sha' to fill missing data.
3. Extracts patch features from diff content.
4. Applies TF-IDF vectorization to textual columns like commit messages.
5. Encodes categorical variables (e.g., smell severity, bug status), saving mappings to JSON for reuse.
6. Cleans the DataFrame: removes columns with too many missing values, encodes non-numeric columns.
7. Saves the transformed dataset to a new CSV file.
"""

import pandas as pd
import os
import sys
import json
import re
import argparse
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import numpy as np

TEXT_COLUMNS = ["message", "bug_message", "code_smell_message"]
COLUMNS_TO_DROP = ['author', 'files', 'url', 'email', 'commit_sha', 'file', 'branch_commits', 'branch_x', 'branch_y', 'branch_sonar', 'commit_date']
METRIC_COLUMNS = [
    'code_smells', 'bugs', 'vulnerabilities', 'coverage',
    'duplicated_lines_density', 'ncloc', 'functions',
    'complexity', 'comment_lines', 'sqale_index', 'sqale_debt_ratio'
]

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder

def apply_label_encoding(df: pd.DataFrame, exclude_columns: List[str]) -> pd.DataFrame:
    categorical_cols = [
        col for col in df.columns 
        if df[col].dtype == object and col not in exclude_columns
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df

def extract_final_identifier(filename: str) -> str:
    match = re.search(r"final_([^_]+)_", filename)
    return match.group(1) if match else "default"

def extract_patch_features(df):
    def process_patch(patch):
        if pd.isna(patch):
            return {
                'patch_size': 0,
                'patch_has_new_test': 0,
                'patch_num_control_flow_changes': 0,
                'patch_comment_lines_added': 0,
                'patch_max_added_line_length': 0,
                'patch_has_exception_handling_change': 0,
            }

        lines = patch.split('\n')
        added_lines = [line for line in lines if line.startswith('+') and not line.startswith('+++')]

        has_new_test = any(('test' in line.lower() or 'assert' in line.lower()) for line in added_lines)
        control_flow_keywords = ['if', 'for', 'while', 'switch', 'case', 'else']
        num_control_flow_changes = sum(any(keyword in line for keyword in control_flow_keywords) for line in added_lines)
        comment_patterns = ['//', '#', '/*', '*', '--']
        comment_lines_added = sum(any(pattern in line for pattern in comment_patterns) for line in added_lines)
        max_added_line_length = max((len(line) for line in added_lines), default=0)
        exception_keywords = ['try', 'catch', 'throw', 'throws', 'except']
        has_exception_handling_change = any(any(keyword in line for keyword in exception_keywords) for line in added_lines)

        return {
            'patch_size': len(lines),
            'patch_has_new_test': int(has_new_test),
            'patch_num_control_flow_changes': num_control_flow_changes,
            'patch_comment_lines_added': comment_lines_added,
            'patch_max_added_line_length': max_added_line_length,
            'patch_has_exception_handling_change': int(has_exception_handling_change),
        }

    patch_features = df['patch'].apply(process_patch)
    patch_features_df = pd.DataFrame(list(patch_features))
    total_changes = df['additions'].fillna(0) + df['deletions'].fillna(0)
    df['patch_additions_ratio'] = df['additions'].fillna(0) / total_changes.replace(0, 1)
    df['patch_deletions_ratio'] = df['deletions'].fillna(0) / total_changes.replace(0, 1)
    df = pd.concat([df.reset_index(drop=True), patch_features_df.reset_index(drop=True)], axis=1)
    return df

def remove_columns_with_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.nunique(dropna=False) > 1]

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

def load_or_create_column_mapping(input_path: str, column_name: str, values: List[str]) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "json_config")
    os.makedirs(config_dir, exist_ok=True)
    base_name = os.path.basename(input_path)
    identifier = extract_final_identifier(base_name)
    json_filename = f"{identifier}__{column_name}.json"
    json_path = os.path.join(config_dir, json_filename)

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            mapping = json.load(f)
    else:
        mapping = {}

    next_code = max(mapping.values(), default=-1) + 1
    for val in values:
        if val not in mapping:
            mapping[val] = next_code
            next_code += 1

    with open(json_path, 'w') as f:
        json.dump(mapping, f, indent=4)

    return mapping

def load_or_create_extension_mapping(input_path: str, extensions: List[str]) -> dict:
    return load_or_create_column_mapping(input_path, 'file_extension', extensions)


def prepare_dataframe(df: pd.DataFrame, input_path: str) -> pd.DataFrame:
    # Garantir que commit_date seja datetime para ordenar corretamente
    if 'commit_date' in df.columns:
        df['commit_date'] = pd.to_datetime(df['commit_date'], utc=True)

    # Propagar métricas por sha
    if 'sha' in df.columns:
        metrics_by_sha = df.dropna(subset=METRIC_COLUMNS).groupby('sha')[METRIC_COLUMNS].first()
        df = df.drop(columns=[col for col in METRIC_COLUMNS if col in df.columns], errors='ignore')
        df = df.merge(metrics_by_sha, on='sha', how='left')

    # Calcular diferença apenas entre SHAs diferentes, ordenando por commit_date
    DIFF_COLUMNS = ['bugs', 'code_smells', 'comment_lines', 'functions']
    if 'sha' in df.columns and all(col in df.columns for col in DIFF_COLUMNS) and 'commit_date' in df.columns:
        # Obter métricas por sha, ordenadas pela data
        sha_ordered = df[['sha', 'commit_date']].drop_duplicates().sort_values('commit_date')
        metrics_ordered = sha_ordered.merge(df.groupby('sha')[DIFF_COLUMNS].first().reset_index(), on='sha')

        # Calcular diferenças entre SHAs consecutivos
        diffs = metrics_ordered[DIFF_COLUMNS].diff().fillna(0).clip(lower=0)  # Zera valores negativos
        diffs['sha'] = metrics_ordered['sha'].values

        # Mapear essas diferenças de volta ao DataFrame original
        for col in DIFF_COLUMNS:
            diff_map = dict(zip(diffs['sha'], diffs[col]))
            df[col] = df['sha'].map(diff_map)

    # Remover colunas desnecessárias
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], errors='ignore')

    # Tratar a extensão do arquivo
    if 'filename' in df.columns:
        df['file_extension'] = df['filename'].astype(str).apply(lambda x: os.path.splitext(x)[1].lower())
        df = df.drop(columns=['filename'])
        extensions = df['file_extension'].unique().tolist()
        ext_mapping = load_or_create_extension_mapping(input_path, extensions)
        df['file_extension'] = df['file_extension'].map(ext_mapping).fillna(-1).astype(int)

    # Codificar colunas categóricas
    categorical_columns = ['smell_severity', 'smell_status', 'bug_severity', 'bug_status']
    for col in categorical_columns:
        if col in df.columns:
            values = df[col].astype(str).unique().tolist()
            mapping = load_or_create_column_mapping(input_path, col, values)
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

    # Aplicar TF-IDF
    df, tfidf_columns = apply_tfidf_to_messages(df)

    # Separar coluna alvo
    if 'failure_prone' in df.columns:
        target_col = df['failure_prone']
        df = df.drop(columns=['failure_prone'])
    else:
        target_col = pd.Series(index=df.index, data=None, name='failure_prone')

    # Remover colunas com valores únicos
    df = remove_columns_with_unique_values(df)

    df = df.loc[:, df.isnull().mean() < 0.7]
    df = df.dropna()

    # Remover sha e patch após uso
    df = df.drop(columns=['sha', 'patch'], errors='ignore')

    # Label encoding
    df = apply_label_encoding(df, exclude_columns=tfidf_columns)

    # Reconectar coluna alvo
    target_col = target_col.loc[df.index]
    df = pd.concat([df, target_col], axis=1)

    return df


def main():
    parser = argparse.ArgumentParser(description='Feature extraction and cleaning of commit CSV.')
    parser.add_argument('csv_file', help='Path to the input CSV file')
    args = parser.parse_args()

    input_path = args.csv_file
    df = pd.read_csv(input_path)
    df = extract_patch_features(df)
    df = prepare_dataframe(df, input_path)

    output_path = input_path.replace('.csv', '_new_features.csv')
    df.to_csv(output_path, index=False)
    print(f'Processed file saved to: {output_path}')


if __name__ == "__main__":
    main()