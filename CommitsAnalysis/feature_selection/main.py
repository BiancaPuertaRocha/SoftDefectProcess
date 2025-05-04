
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import argparse

from ga import GAFeatureSelector
from fisher_score import FisherScoreFeatureSelector

def remove_columns_with_unique_values(df):
    df_cleaned = df.loc[:, df.nunique() > 1]
    return df_cleaned


def run_fisher_random_forest(df, csv_filename):
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    fisher_feature_selector = FisherScoreFeatureSelector(clf)
    selected_features = fisher_feature_selector.run(df)
    
    print(f"Selected Features: {selected_features}")

    selected_df = df[selected_features]

    function_name = "run_fisher_random_forest"
    filename = f"{csv_filename}_{function_name}_selected_features.csv"
    selected_df.to_csv(filename, index=False)

    print(f"Selected features saved to: {filename}")

def consider_messages(df, max_features=20):
    df['bug_message'] = df['bug_message'].fillna('')
    df['message'] = df['message'].fillna('')
    df['code_smell_message'] = df['code_smell_message'].fillna('')

    # Apply TF-IDF
    tfidf_bug_message = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_message = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_code_smell_message = TfidfVectorizer(stop_words='english', max_features=max_features)

    X_bug_message = tfidf_bug_message.fit_transform(df['bug_message']).toarray()
    X_message = tfidf_message.fit_transform(df['message']).toarray()
    X_code_smell_message = tfidf_code_smell_message.fit_transform(df['code_smell_message']).toarray()

    bug_message_vocab = tfidf_bug_message.get_feature_names_out()
    message_vocab = tfidf_message.get_feature_names_out()
    code_smell_message_vocab = tfidf_code_smell_message.get_feature_names_out()

    # create dfs with the word cols
    df_bug_message = pd.DataFrame(X_bug_message, columns=[f'bug_message_{word}' for word in bug_message_vocab])
    df_message = pd.DataFrame(X_message, columns=[f'message_{word}' for word in message_vocab])
    df_code_smell_message = pd.DataFrame(X_code_smell_message, columns=[f'code_smell_message_{word}' for word in code_smell_message_vocab])

    # join word columns 
    df = df.reset_index(drop=True)
    df_bug_message = df_bug_message.reset_index(drop=True)
    df_message = df_message.reset_index(drop=True)
    df_code_smell_message = df_code_smell_message.reset_index(drop=True)

    df_final = pd.concat([df, df_bug_message, df_message, df_code_smell_message], axis=1)

    return df_final


def process_csv_and_run_function(csv_filename, function_name):
    all_chunks = []
    colunas_extra = ["message", "bug_message", "code_smell_message"]

    for chunk in pd.read_csv(csv_filename, chunksize=10000):
        # Transform categoric to numeric
        if 'bug_status' in chunk.columns and 'smell_status' in chunk.columns:
            chunk['bug_status'] = chunk['bug_status'].astype('category').cat.codes
            chunk['smell_status'] = chunk['smell_status'].astype('category').cat.codes

        # Seleciona colunas numéricas
        df_numeric = chunk.select_dtypes(include='number')

        # Seleciona apenas as colunas extras que existem no chunk
        colunas_existentes = [col for col in colunas_extra if col in chunk.columns]

        # Junta numéricos + colunas de texto
        df_final_chunk = pd.concat([df_numeric, chunk[colunas_existentes]], axis=1)

        print(f"Colunas do pedaço: {list(df_final_chunk.columns)}")

        all_chunks.append(df_final_chunk)

    # Junta todos os pedaços em um único DataFrame
    df_final = pd.concat(all_chunks, ignore_index=True)

    # Executa a função consider_messages, se definida
    df_final = consider_messages(df_final)
    df_final = remove_columns_with_unique_values(df_final)
    
    df_final = df_final.drop(columns=colunas_extra)

    print(f"\nDataFrame: {len(df_final)} linhas.")

    # Chama a função de seleção de atributos, se existir
    if function_name in globals():
        print(f"Executando função '{function_name}'")
        globals()[function_name](df_final, csv_filename)
    else:
        print(f"Função '{function_name}' não encontrada.")


def main():
    parser = argparse.ArgumentParser(description="Process CSV an run FS function")
    parser.add_argument("csv_filename", help="Name of CSV file")
    parser.add_argument("function_name", help="Function name run_ga_random_forest, run_chi2_random_forest)")
    
    args = parser.parse_args()

    process_csv_and_run_function(args.csv_filename, args.function_name)

if __name__ == "__main__":
    main()
