'''
APLICAÇÃO DO SZZ, VERIFICANDO OS ARQUIVOS QUE FORAM MODIFICADOS NO COMMIT E QUE FORAM POSTERIORMENTE MODIFICADOS EM UM PULL
REQUEST DE UMA ISSUE, PARA VERIFICAR SE ESSE COMMIT É BUG INDUCING
'''

import pandas as pd
import sqlite3

# Substitua pelos caminhos corretos
csv_file_path = 'commits_data_dubbo.csv'
db_file_path = 'SDP1 - dubbo complete.db'
bug_introducing_commits_csv = 'bug_introducing_commits.csv'
output_final_csv = 'final_metrics_dubbo.csv'

# Carregar o CSV inicial em um DataFrame
df_commits = pd.read_csv(csv_file_path)

# Conectar ao banco de dados SQLite
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# Criar a tabela commits no SQLite com a nova estrutura
cursor.execute('''
CREATE TABLE IF NOT EXISTS commits (
    sha TEXT PRIMARY KEY,
    author TEXT,
    email TEXT,
    message TEXT,
    commit_date DATETIME,
    url TEXT,
    files_changed INT,
    modified_files TEXT,
    additions INT,
    deletions INT,
    total_changes INT
)
''')
conn.commit()

# Inserir os dados do DataFrame na tabela commits
df_commits.to_sql('commits', conn, if_exists='replace', index=False)


# Relacionar os arquivos modificados nos commits com os arquivos modificados nos pull requests
query_related_commits = '''
SELECT 
    c.sha AS "commit", 
    pfc.FILE_NAME AS filepath, 
    p.PR_ID, 
    p.ISSUE_ID
FROM 
    commits c
JOIN 
    pr_files_changed pfc ON ',' || c.modified_files || ',' LIKE '%,' || pfc.FILE_NAME || ',%'
JOIN 
    PULL_RQ p ON pfc.PR_NO = p.PR_ID
WHERE 
    p.ISSUE_ID IS NOT NULL;
'''

df_related_commits = pd.read_sql_query(query_related_commits, conn)

# Identificar o commit que introduziu o bug
query_bug_introducing_commits = '''
WITH IssueCommits AS (
    SELECT 
        i.ISSUE_ID, 
        i.OPEN_DATE, 
        p.PR_ID, 
        c.sha AS "commit", 
        pfc.FILE_NAME AS filepath,
        c.commit_date AS committed_at
    FROM 
        ISSUE i
    JOIN 
        PULL_RQ p ON i.ISSUE_ID = p.ISSUE_ID
    JOIN 
        pr_files_changed pfc ON p.PR_ID = pfc.PR_NO
    JOIN 
        commits c ON ',' || c.modified_files || ',' LIKE '%,' || pfc.FILE_NAME || ',%'
    WHERE 
        c.commit_date < i.OPEN_DATE
)
SELECT 
    ic.ISSUE_ID, 
    ic.PR_ID, 
    ic."commit", 
    ic.filepath,
    ic.committed_at
FROM 
    IssueCommits ic
JOIN 
    (SELECT ISSUE_ID, MAX(committed_at) as last_commit_date FROM IssueCommits GROUP BY ISSUE_ID) ic_max 
    ON ic.ISSUE_ID = ic_max.ISSUE_ID AND ic.committed_at = ic_max.last_commit_date;
'''


df_bug_introducing_commits = pd.read_sql_query(query_bug_introducing_commits, conn)

# Fechar a conexão com o banco de dados
conn.close()

# Salvar o resultado em um arquivo CSV
df_bug_introducing_commits.to_csv(bug_introducing_commits_csv, index=False)

# Carregar o CSV de commits que introduziram bugs
df_bug_introducing_commits = pd.read_csv(bug_introducing_commits_csv)

# Atualizar a coluna failure_prone no DataFrame inicial
df_commits['failure_prone'] = df_commits['sha'].isin(df_bug_introducing_commits['commit'])

# Salvar o DataFrame atualizado em um novo arquivo CSV
df_commits.to_csv(output_final_csv, index=False)

print(f"Resultado salvo em {output_final_csv}")
