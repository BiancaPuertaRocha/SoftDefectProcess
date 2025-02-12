import requests
import csv
import time
import argparse

# GitHub Token for authentication
GITHUB_TOKEN = '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'

# Function to fetch commit details from GitHub API
def fetch_commit_details(base_url, commit_sha, headers):
    url = f'{base_url}/{commit_sha}'
    failed_attempts = 0  # Contador de falhas seguidas

    while True:  # Loop infinito para tentar indefinidamente
        try:
            response = requests.get(url, headers=headers, timeout=10)  # Timeout para evitar travamentos
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:  # Rate limit exceeded
                print(f'Rate limit reached. Waiting 60 seconds...')
                time.sleep(60)
            else:
                print(f'Error accessing API for commit {commit_sha}: {response.status_code} - {response.text}')
                return None
        except Exception as e:
            print(f'Connection error: {e}. Retrying...')
            time.sleep(5)  # Espera um tempo antes de tentar novamente
            failed_attempts += 1

            if failed_attempts >= 3:  # Se falhar várias vezes seguidas, tenta uma pausa maior
                print("Failed multiple times in a row. Waiting 1 hour before retrying...")
                time.sleep(3600)  # Espera 1 hora antes de tentar novamente
                failed_attempts = 0  # Reseta o contador

# Function to save commit data to a CSV file
def save_commit_to_csv(output_csv, commit_data):
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sha', 'branch', 'author', 'email', 'message', 'commit_date', 'url', 'files_changed', 'modified_files', 'additions', 'deletions', 'total_changes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(commit_data)

# Function to fetch branches from GitHub API
def fetch_branches(repo_owner, repo_name, headers):
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/branches'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return [branch['name'] for branch in response.json()]
    else:
        print(f'Error accessing branches: {response.status_code} - {response.text}')
        return []

def main():
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description='Fetch commits from a GitHub repository and save to CSV.')
    parser.add_argument('--repo_owner', type=str, required=True, help='Owner of the GitHub repository')
    parser.add_argument('--repo_name', type=str, required=True, help='Name of the GitHub repository')
    parser.add_argument('--output_csv', type=str, required=True, help='Name of the output CSV file')
    args = parser.parse_args()

    # Configuration
    repo_owner = args.repo_owner
    repo_name = args.repo_name
    output_csv = args.output_csv

    # GitHub API base URL for commits
    base_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/commits'

    # Request headers with authentication token
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }

    commits_saved = 0
    requests_made = 0

    # Open CSV file and write header
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sha', 'branch', 'author', 'email', 'message', 'commit_date', 'url', 'files_changed', 'modified_files', 'additions', 'deletions', 'total_changes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Fetch all branches
    branches = fetch_branches(repo_owner, repo_name, headers)
    if not branches:
        print(f'No branches found for the repository: {repo_name}')
        return

    # Iterate over each branch to get commits
    for branch in branches:
        print(f'Fetching commits for branch: {branch}')
        page = 1
        while True:
            url = f'{base_url}?sha={branch}&page={page}&per_page=100'  
            response = requests.get(url, headers=headers)
            requests_made += 1

            if response.status_code == 200:
                commits = response.json()
                if len(commits) == 0:
                    break
                for commit in commits:
                    commit_sha = commit['sha']

                    commit_details = fetch_commit_details(base_url, commit_sha, headers)
                    if commit_details:
                        commit_data = {
                            'sha': commit_details['sha'],
                            'branch': branch,  # Adding the branch to the data
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
                        save_commit_to_csv(output_csv, commit_data)
                        commits_saved += 1

                    time.sleep(0.5)

                    if commits_saved % 10 == 0:
                        print(f'Commits saved so far: {commits_saved}')
                        print(f'Requests made so far: {requests_made}')
                        print('---')

                page += 1
            elif response.status_code == 403:  # Rate limit exceeded
                print(f'Rate limit reached. Waiting 60 seconds...')
                time.sleep(60)
            else:
                print(f'Error accessing API: {response.status_code} - {response.text}')
                break

    print(f'Commit data saved in {output_csv}')
    print(f'Total commits saved: {commits_saved}')
    print(f'Total requests made: {requests_made}')

if __name__ == '__main__':
    main()
