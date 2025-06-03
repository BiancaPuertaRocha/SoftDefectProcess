import pandas as pd
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_COLUMNS = ["message", "bug_message", "code_smell_message"]

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

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Codifica colunas categóricas específicas, se existirem
    for label_col in ['bug_status', 'smell_status']:
        if label_col in df.columns:
            df[label_col] = df[label_col].astype('category').cat.codes

    # Aplica TF-IDF nas colunas de texto
    df, tfidf_columns = apply_tfidf_to_messages(df)

    # Separa a coluna alvo, se existir
    if 'failure_prone' in df.columns:
        target_col = df['failure_prone']
        df = df.drop(columns=['failure_prone'])
    else:
        target_col = pd.Series(index=df.index, data=None, name='failure_prone')

    # Identifica colunas numéricas e não-TFIDF
    non_tfidf_columns = [col for col in df.columns if col not in tfidf_columns]

    # Remove colunas com mais de 50% de valores nulos
    cols_to_drop = [col for col in non_tfidf_columns if df[col].isna().mean() > 0.5]
    df = df.drop(columns=cols_to_drop)

    # Remove linhas com NaNs nas colunas restantes
    cols_to_check = [col for col in non_tfidf_columns if col not in cols_to_drop]
    df = df.dropna(subset=cols_to_check)

    # Junta novamente a variável alvo
    target_col = target_col.loc[df.index]
    df = pd.concat([df, target_col], axis=1)

    # Remove colunas com apenas um valor único
    df = remove_columns_with_unique_values(df)

    return df

def main():
    input_path = sys.argv[1]

    # Carrega o CSV
    df = pd.read_csv(input_path)

    # Aplica extração de features do patch
    df = extract_patch_features(df)

    # Aplica TF-IDF nas colunas de texto, transforma em numerico
    df = prepare_dataframe(df)

    # Gera o novo caminho para salvar
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_name = f"{name}_new_features{ext}"
    output_path = os.path.join(dir_name, output_name)

    # Salva o novo CSV
    df.to_csv(output_path, index=False)

    print(f"Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    main()
