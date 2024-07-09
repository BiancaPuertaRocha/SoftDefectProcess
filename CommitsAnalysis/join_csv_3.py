'''
COMPILAR COMMITS COM A COLUNA A MAIS DE PROPENSAO A ERROS
'''

import pandas as pd

# Carregar os CSVs
df1 = pd.read_csv('commits_data_dubbo.csv')
df2 = pd.read_csv('bug_introducing_commits.csv')

# Adicionar a coluna 'failure_prone' ao df1
df1['failure_prone'] = df1['sha'].apply(lambda x: 1 if x in df2['commit'].values else 0)

# Salvar o novo CSV
df1.to_csv('novo.csv', index=False)

# Verificar o resultado
print(df1.head())