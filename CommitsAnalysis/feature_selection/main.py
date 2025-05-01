
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

# Função exemplo: run_ga_random_forest
def run_ga_random_forest(df):
    # Aqui você pode colocar a lógica da sua função
    print("Executando função: run_ga_random_forest")
    print(df.head())
    print(f"colunas {len(df.columns)}")


def consider_messages(df):
    df['bug_message'] = df['bug_message'].fillna('')
    df['message'] = df['message'].fillna('')
    df['code_smell_message'] = df['code_smell_message'].fillna('')

    # Vetorizadores TF-IDF
    tfidf_bug_message = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_message = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_code_smell_message = TfidfVectorizer(stop_words='english', max_features=max_features)

    # Aplicação do TF-IDF
    X_bug_message = tfidf_bug_message.fit_transform(df['bug_message']).toarray()
    X_message = tfidf_message.fit_transform(df['message']).toarray()
    X_code_smell_message = tfidf_code_smell_message.fit_transform(df['code_smell_message']).toarray()

    # Recuperar vocabulário
    bug_message_vocab = tfidf_bug_message.get_feature_names_out()
    message_vocab = tfidf_message.get_feature_names_out()
    code_smell_message_vocab = tfidf_code_smell_message.get_feature_names_out()

    # Criar DataFrames com as novas colunas
    df_bug_message = pd.DataFrame(X_bug_message, columns=[f'bug_message_{word}' for word in bug_message_vocab])
    df_message = pd.DataFrame(X_message, columns=[f'message_{word}' for word in message_vocab])
    df_code_smell_message = pd.DataFrame(X_code_smell_message, columns=[f'code_smell_message_{word}' for word in code_smell_message_vocab])

    # Resetar índice para garantir concatenação correta
    df = df.reset_index(drop=True)
    df_bug_message = df_bug_message.reset_index(drop=True)
    df_message = df_message.reset_index(drop=True)
    df_code_smell_message = df_code_smell_message.reset_index(drop=True)

    # Concatenar com o DataFrame original
    df_final = pd.concat([df, df_bug_message, df_message, df_code_smell_message], axis=1)

    return df_final


def process_csv_and_run_function(csv_filename, function_name):
    numeric_chunks = []

    for chunk in pd.read_csv(csv_filename, chunksize=10000):
        # Transformações nas colunas categóricas, se existirem
        if 'bug_status' in chunk.columns and 'smell_status' in chunk.columns:
            chunk['bug_status'] = chunk['bug_status'].astype('category').cat.codes
            chunk['smell_status'] = chunk['smell_status'].astype('category').cat.codes

        # Seleciona colunas numéricas
        df_numeric = chunk.select_dtypes(include='number')
        print(f"Colunas numéricas do pedaço: {list(df_numeric.columns)}")

        numeric_chunks.append(df_numeric)

    # Concatena todos os pedaços numéricos
    df_final = pd.concat(numeric_chunks, ignore_index=True)
    print(f"\nDataFrame final com {len(df_final)} linhas.")

    df_final = consider_messages(df)
    # Chama a função, se estiver definida
    if function_name in globals():
        print(f"Executando função '{function_name}' com o DataFrame final...")
        globals()[function_name](df_final)
    else:
        print(f"Função '{function_name}' não foi encontrada.")


def main():
    # Configuração para ler os argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Processar CSV e executar a função especificada")
    parser.add_argument("csv_filename", help="Nome do arquivo CSV")
    parser.add_argument("function_name", help="Nome da função a ser executada")
    
    args = parser.parse_args()

    # Processa o CSV e chama a função especificada
    process_csv_and_run_function(args.csv_filename, args.function_name)

if __name__ == "__main__":
    main()
