import pandas as pd
import sqlite3

repo = 'dubbo'

csv_file_path = f'commits_from_api/commits_data_{repo}.csv'
db_file_path = f'issues_by_sdptool/SDP1 - {repo} complete.db'
issues_commits_from_repo = f'issues_commits/issues_commits_{repo}.csv'
output_final_csv = f'final/final_metrics_{repo}.csv'
other_csv_path = f'results_sonar/sonarqube_metrics_{repo}.csv'  # Insira o caminho do outro CSV aqui

# Carregar o CSV inicial em um DataFrame
df_commits = pd.read_csv(csv_file_path)

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
conn.close()

df_bug_introducing_commits.to_csv(issues_commits_from_repo, index=False)

# Carregar o CSV de commits que introduziram bugs
df_bug_introducing_commits = pd.read_csv(issues_commits_from_repo)

# Atualizar a coluna failure_prone no DataFrame inicial
df_commits['failure_prone'] = df_commits['sha'].isin(df_bug_introducing_commits['commit'])

# Carregar o outro CSV que contém o atributo commit_sha
df_other = pd.read_csv(other_csv_path)

# Realizar a junção com base no atributo commit_sha
df_final = df_commits.merge(df_other, left_on='sha', right_on='commit_sha', how='left')

# Salvar o DataFrame final em um novo arquivo CSV
df_final.to_csv(output_final_csv, index=False)

print(f"Resultado salvo em {output_final_csv}")
