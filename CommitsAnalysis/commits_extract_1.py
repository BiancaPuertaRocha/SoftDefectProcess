'''
EXTRAÇÃO DE COMMITS VIA API DO GITHUB APÓS A EXTRAÇÃO DAS ISSUES, PULL RQUESTS E FILES DOS PULL REQUESTS 
EXTRAIDOS VIA SDPTOOL
'''


import requests
import csv
import time

# Configurações
github_token = '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'  # Substitua pelo seu token de autenticação
repo_owner = 'apache'
repo_name = 'airflow'
output_csv = f'commits_data_{repo_name}.csv'  # Nome do arquivo CSV de saída

head = False
# URL base da API do GitHub para commits
base_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/commits'

# Cabeçalhos da requisição com o token de autenticação
headers = {
    'Authorization': f'token {github_token}',
    'Accept': 'application/vnd.github.v3+json'
}

# Função para fazer requisição à API do GitHub e retornar dados de um commit específico
def fetch_commit_details(commit_sha):
    url = f'{base_url}/{commit_sha}'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 403:  
        print(f'Limite de requisições atingido. Esperando 60 segundos...')
        time.sleep(60)
        return fetch_commit_details(commit_sha)  
    else:
        print(f'Erro ao acessar a API para o commit {commit_sha}: {response.status_code} - {response.text}')
        return None

# Função para salvar dados de um commit em um arquivo CSV
def save_commit_to_csv(commit_data):
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sha', 'author', 'email', 'message', 'commit_date', 'url', 'files_changed', 'modified_files', 'additions', 'deletions', 'total_changes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(commit_data)


commits_saved = 0
requests_made = 0

# Paginação da API do GitHub para buscar todos os commits
page = 1
with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['sha', 'author', 'email', 'message', 'commit_date', 'url', 'files_changed', 'modified_files', 'additions', 'deletions', 'total_changes']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

while True:
    url = f'{base_url}?page={page}&per_page=100'  
    response = requests.get(url, headers=headers)
    requests_made += 1 
    if response.status_code == 200:
        commits = response.json()
        if len(commits) == 0:
            break
        for commit in commits:
            commit_sha = commit['sha']

            commit_details = fetch_commit_details(commit_sha)
            if commit_details:
                commit_data = {
                    'sha': commit_details['sha'],
                    'author': commit_details['commit']['author']['name'],
                    'email': commit_details['commit']['author']['email'],
                    'message': commit_details['commit']['message'],
                    'commit_date': commit_details['commit']['author']['date'], 
                    'url': commit_details['html_url'], 
                    'files_changed': len(commit_details['files']) if 'files' in commit_details else 0, 
                    'modified_files': ','.join([file['filename'] for file in commit_details['files']]) if 'files' in commit_details else '', 
                    'additions': commit_details['stats']['additions'] if 'stats' in commit_details else 0, 
                    'deletions': commit_details['stats']['deletions'] if 'stats' in commit_details else 0,  
                    'total_changes': commit_details['stats']['total'] if 'stats' in commit_details else 0,  
                }
                save_commit_to_csv(commit_data)
                commits_saved += 1  
            else:
                print(f'Não foi possível obter detalhes para o commit {commit_sha}')
                
            time.sleep(0.5) 

            if commits_saved % 10 == 0:
                print(f'Commits salvos até agora: {commits_saved}')
                print(f'Requisições feitas até agora: {requests_made}')
                print('---')

        page += 1
    elif response.status_code == 403:  
        print(f'Limite de requisições atingido. Esperando 60 segundos...')
        time.sleep(60)
    else:
        print(f'Erro ao acessar a API: {response.status_code} - {response.text}')
        break


print(f'Dados de commits salvos em {output_csv}')
print(f'Commits salvos: {commits_saved}')
print(f'Requisições feitas: {requests_made}')
