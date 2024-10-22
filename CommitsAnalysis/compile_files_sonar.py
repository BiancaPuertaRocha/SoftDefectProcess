import pandas as pd
import argparse
import os

# Configurar argparse para receber o prefixo e a pasta
parser = argparse.ArgumentParser(description='Combine bug, smell, and SonarQube metric files.')
parser.add_argument('prefix', type=str, help='Prefixo do arquivo (ex: "dubbo")')
parser.add_argument('directory', type=str, help='Caminho para o diretório contendo os arquivos')

args = parser.parse_args()

# Usar os argumentos fornecidos
prefix = args.prefix
directory = args.directory

# Construir os caminhos completos dos arquivos com base no prefixo e na pasta
bugs_file = os.path.join(directory, f'bugs_{prefix}_final.csv')
smells_file = os.path.join(directory, f'smells_{prefix}_final.csv')
metrics_file = os.path.join(directory, f'sonarqube_metrics_{prefix}_final.csv')

# Carregar os arquivos CSV
bugs_df = pd.read_csv(bugs_file)
smells_df = pd.read_csv(smells_file)
metrics_df = pd.read_csv(metrics_file)

# Renomear colunas repetidas para identificar suas origens
bugs_df = bugs_df.rename(columns={'line': 'bug_line', 'severity': 'bug_severity', 'status': 'bug_status'})
smells_df = smells_df.rename(columns={'line': 'smell_line', 'severity': 'smell_severity', 'status': 'smell_status'})

# Unir os DataFrames por 'commit_sha' e 'file' (chaves compostas)
merged_df = pd.merge(bugs_df, smells_df, on=['commit_sha', 'file'], how='outer')
merged_df = pd.merge(merged_df, metrics_df, on=['commit_sha'], how='outer')

# Salvar o resultado final em um novo arquivo CSV
output_file = os.path.join(directory, f'merged_modifications_{prefix}_final.csv')
merged_df.to_csv(output_file, index=False)

print(f"Arquivo '{output_file}' compilado com sucesso.")
