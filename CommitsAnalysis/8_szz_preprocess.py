import pandas as pd
import argparse

# Colunas a remover
COLUMNS_TO_DROP = ['author', 'files', 'url', 'email', 'file', 'branch_commits', 'branch_x', 'branch_y', 'branch_sonar', 'commit_sha']
# Métricas a serem propagadas
METRIC_COLUMNS = [
    'code_smells', 'bugs', 'vulnerabilities', 'coverage',
    'duplicated_lines_density', 'ncloc', 'functions',
    'complexity', 'comment_lines', 'sqale_index', 'sqale_debt_ratio'
]

def main():
    parser = argparse.ArgumentParser(description='Limpa e propaga métricas por sha.')
    parser.add_argument('csv_file', help='Caminho para o arquivo CSV')
    args = parser.parse_args()

    # Leitura do CSV
    df = pd.read_csv(args.csv_file)

    # Remove colunas se existirem
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], errors='ignore')

    # Preenche métricas com base no sha
    metrics_by_sha = df.dropna(subset=METRIC_COLUMNS).groupby('sha')[METRIC_COLUMNS].first()
    df = df.drop(columns=[col for col in METRIC_COLUMNS if col in df.columns], errors='ignore')
    df = df.merge(metrics_by_sha, on='sha', how='left')

    # Salva o novo CSV
    output_path = args.csv_file.replace('.csv', '_clear.csv')
    df.to_csv(output_path, index=False)
    print(f'Arquivo processado salvo em: {output_path}')

if __name__ == '__main__':
    main()
