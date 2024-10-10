import git
import subprocess
import json
import os
import csv
import requests
import argparse
from utils import generate_new_sonar_token

SONAR_SCANNER_CMD = 'sonar-scanner'
SONAR_URL = 'http://localhost:9001'
SONAR_USER = 'admin'  # SonarQube user
SONAR_PASS = 'admin'  # SonarQube pass
SONAR_TOKEN = None    # Auto generated token
RESULTS_DIR = 'results_sonar'

def get_sonar_metrics(sonar_project_key, commit_sha):
    global SONAR_TOKEN
    metric_keys = 'code_smells,bugs,vulnerabilities,coverage,duplicated_lines_density,ncloc,files,functions,complexity,comment_lines,sqale_index,sqale_debt_ratio'
    measures_url = f'{SONAR_URL}/api/measures/component'
    params = {
        'component': sonar_project_key,
        'metricKeys': metric_keys,
    }
    auth = (SONAR_TOKEN, '')

    response = requests.get(measures_url, params=params, auth=auth)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:  # Auth failed, regenerate token
        SONAR_TOKEN = generate_new_sonar_token(sonar_url=SONAR_URL, sonar_user=SONAR_USER, sonar_pass=SONAR_PASS)
        auth = (SONAR_TOKEN, '')
        response = requests.get(measures_url, params=params, auth=auth)
        if response.status_code == 200:
            return response.json()
    print(f"Erro ao obter métricas do SonarQube: {response.text}")
    return None

def get_bug_locations(sonar_project_key):
    issues_url = f'{SONAR_URL}/api/issues/search'
    params = {
        'componentKeys': sonar_project_key,
        'types': 'BUG',
        'statuses': 'OPEN,REOPENED,CONFIRMED',
        'resolved': 'false'
    }
    auth = (SONAR_TOKEN, '')
    response = requests.get(issues_url, params=params, auth=auth)
    if response.status_code == 200:
        return response.json()
    print(f"Erro ao obter bugs do SonarQube: {response.text}")
    return None

def get_code_smells(sonar_project_key):
    issues_url = f'{SONAR_URL}/api/issues/search'
    params = {
        'componentKeys': sonar_project_key,
        'types': 'CODE_SMELL',
        'statuses': 'OPEN,REOPENED,CONFIRMED',
        'resolved': 'false'
    }
    auth = (SONAR_TOKEN, '')
    response = requests.get(issues_url, params=params, auth=auth)
    if response.status_code == 200:
        return response.json()
    print(f"Erro ao obter code smells do SonarQube: {response.text}")
    return None

def save_bugs_to_csv(bugs_data, commit_sha, initial_dir, output_csv_bugs):
    result_file = os.path.join(initial_dir, RESULTS_DIR, output_csv_bugs)
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as csvfile:
        fieldnames = ['commit_sha', 'file', 'line', 'bug_message', 'severity', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for issue in bugs_data['issues']:
            locations = issue.get('textRange', {})
            file_info = {
                'commit_sha': commit_sha,
                'file': issue['component'],
                'line': locations.get('startLine', 'N/A'),
                'bug_message': issue['message'],
                'severity': issue['severity'],
                'status': issue['status']
            }
            writer.writerow(file_info)

def save_code_smells_to_csv(code_smells_data, commit_sha, initial_dir, output_csv_smells):
    result_file = os.path.join(initial_dir, RESULTS_DIR, output_csv_smells)
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as csvfile:
        fieldnames = ['commit_sha', 'file', 'line', 'code_smell_message', 'severity', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for issue in code_smells_data['issues']:
            locations = issue.get('textRange', {})
            file_info = {
                'commit_sha': commit_sha,
                'file': issue['component'],
                'line': locations.get('startLine', 'N/A'),
                'code_smell_message': issue['message'],
                'severity': issue['severity'],
                'status': issue['status']
            }
            writer.writerow(file_info)

def save_metrics_to_csv(metrics_data, commit_sha, initial_dir, output_csv):
    result_file = os.path.join(initial_dir, RESULTS_DIR, output_csv)
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as csvfile:
        fieldnames = ['commit_sha', 'code_smells', 'bugs', 'vulnerabilities', 'coverage', 'duplicated_lines_density', 'ncloc', 'files', 'functions', 'complexity', 'comment_lines', 'sqale_index', 'sqale_debt_ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        measures = metrics_data['component']['measures']
        metrics = {measure['metric']: measure['value'] for measure in measures}
        metrics['commit_sha'] = commit_sha
        writer.writerow(metrics)

def run_sonar_scanner(commit_sha, sonar_project_key, output_csv, output_csv_bugs, output_csv_smells, sonar_sources):
    global SONAR_TOKEN
    repo_path = os.path.join(os.getcwd(), 'repos', sonar_project_key)
    
    initial_dir = os.getcwd()
    os.chdir(repo_path)
    
    try:
        # Checkout specific commit
        repo = git.Repo(repo_path)
        repo.git.checkout(commit_sha)

        sonar_properties = f"""
        sonar.projectKey={sonar_project_key}
        sonar.host.url={SONAR_URL}
        sonar.login={SONAR_TOKEN}
        sonar.sources={sonar_sources},
        sonar.java.binaries=.
        """
        
        with open('sonar-project.properties', 'w') as f:
            f.write(sonar_properties)
        
        # Run Sonar Scanner
        subprocess.run([SONAR_SCANNER_CMD], capture_output=True, text=True)

        # Collect and save metrics
        metrics_data = get_sonar_metrics(sonar_project_key, commit_sha)
        if metrics_data:
            save_metrics_to_csv(metrics_data, commit_sha, initial_dir, output_csv)
        
        # Collect and save bugs
        bugs_data = get_bug_locations(sonar_project_key)
        if bugs_data:
            save_bugs_to_csv(bugs_data, commit_sha, initial_dir, output_csv_bugs)

        # Collect and save code smells
        code_smells_data = get_code_smells(sonar_project_key)
        if code_smells_data:
            save_code_smells_to_csv(code_smells_data, commit_sha, initial_dir, output_csv_smells)

    finally:
        os.chdir(initial_dir)

def main(sonar_project_key, output_csv, output_csv_bugs, output_csv_smells, branch, sonar_sources):
    repo_path = os.path.join(os.getcwd(), 'repos', sonar_project_key)
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if not os.path.exists(repo_path):
        print(f"Repo not found in {repo_path}. Cloning...")
        git.Repo.clone_from('REPO_URL', repo_path)

    repo = git.Repo(repo_path)

    # Run SonarQube analysis for each commit
    for commit in repo.iter_commits(branch):
        print(f"Processing commit {commit.hexsha}...")
        run_sonar_scanner(commit.hexsha, sonar_project_key, output_csv, output_csv_bugs, output_csv_smells, sonar_sources)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SonarQube analysis for all commits in a branch')
    parser.add_argument('--sonar_project_key', required=True, help='SonarQube project key')
    parser.add_argument('--branch', required=True, help='Branch to analyze')
    parser.add_argument('--output_csv', default='sonar_metrics.csv', help='CSV file to save SonarQube metrics')
    parser.add_argument('--output_csv_bugs', default='sonar_bugs.csv', help='CSV file to save bug information')
    parser.add_argument('--output_csv_smells', default='sonar_code_smells.csv', help='CSV file to save code smell information')
    parser.add_argument('--sonar_sources', required=True, help='Path to source files for SonarQube')

    args = parser.parse_args()
    main(args.sonar_project_key, args.output_csv, args.output_csv_bugs, args.output_csv_smells, args.branch, args.sonar_sources)
