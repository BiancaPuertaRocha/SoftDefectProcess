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
    print(f"SonarQube API response status: {response.status_code}")
    print(f"SonarQube API response: {response.text}")
    
    if response.status_code == 401:  # auth failed
        print("Token expired or invalid. Trying to generate a new token...")
        SONAR_TOKEN = generate_new_sonar_token(sonar_url=SONAR_URL, sonar_user=SONAR_USER, sonar_pass=SONAR_PASS)
        if SONAR_TOKEN:
            auth = (SONAR_TOKEN, '')
            response = requests.get(measures_url, params=params, auth=auth)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting metrics with token: {response.text}")
                return None
        else:
            print("Fail generating token.")
            return None
    elif response.status_code == 200:
        return response.json()
    else:
        print(f"Erro ao obter métricas do SonarQube: {response.text}")
        return None


def get_bug_locations(sonar_project_key, commit_sha):
    issues_url = f'{SONAR_URL}/api/issues/search'
    params = {
        'componentKeys': sonar_project_key,
        'types': 'BUG',
        'statuses': 'OPEN,REOPENED,CONFIRMED',
        'resolved': 'false'
    }
    auth = (SONAR_TOKEN, '')

    response = requests.get(issues_url, params=params, auth=auth)
    print(f"SonarQube API issues response status: {response.status_code}")
    print(f"SonarQube API issues response: {response.text}")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erro ao obter issues do SonarQube: {response.text}")
        return None


def save_bugs_to_csv(bugs_data, commit_sha, initial_dir, output_csv_bugs):
    result_file = os.path.join(initial_dir, RESULTS_DIR, output_csv_bugs)
    file_exists = os.path.isfile(result_file)

    print(f"CSV file for bugs exists: {file_exists}")
    with open(result_file, 'a', newline='') as csvfile:
        fieldnames = ['commit_sha', 'file', 'line', 'bug_message', 'severity', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            print(f"CSV header written for bugs: {fieldnames}")
        
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
            print(f"Bug data to write: {file_info}")
            writer.writerow(file_info)
        print(f"Bug data for commit {commit_sha} written to CSV")


def save_metrics_to_csv(metrics_data, commit_sha, initial_dir, output_csv):
    result_file = os.path.join(initial_dir, RESULTS_DIR, output_csv)
    file_exists = os.path.isfile(result_file)

    print(f"CSV file exists: {file_exists}")
    with open(result_file, 'a', newline='') as csvfile:
        fieldnames = ['commit_sha', 'code_smells', 'bugs', 'vulnerabilities', 'coverage', 'duplicated_lines_density', 'ncloc', 'files', 'functions', 'complexity', 'comment_lines', 'sqale_index', 'sqale_debt_ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            print(f"CSV header written: {fieldnames}")
        
        measures = metrics_data['component']['measures']
        metrics = {measure['metric']: measure['value'] for measure in measures}
        metrics['commit_sha'] = commit_sha
        print(f"Metrics to write: {metrics}")
        writer.writerow(metrics)
        print(f"Metrics for commit {commit_sha} written to CSV")


def run_sonar_scanner(commit_sha, sonar_project_key, output_csv, output_csv_bugs, branch, sonar_sources):
    global SONAR_TOKEN
    repo_path = os.path.join(os.getcwd(), 'repos', sonar_project_key)
    
    if not os.path.exists(repo_path):
        print(f"Repo not found in {repo_path}. Cloning...")
        repo = git.Repo.clone_from('REPO_URL', repo_path)
    else:
        repo = git.Repo(repo_path)

    initial_dir = os.getcwd()
    os.chdir(repo_path)
    
    try:
        repo.git.checkout(commit_sha)

        sonar_properties = f"""
        sonar.projectKey={sonar_project_key}
        sonar.host.url={SONAR_URL}
        sonar.login={SONAR_TOKEN}
        sonar.sources={sonar_sources},
        sonar.java.binaries=.
        """
        print(sonar_properties)
        
        with open('sonar-project.properties', 'w') as f:
            f.write(sonar_properties)
        
        result = subprocess.run([SONAR_SCANNER_CMD], capture_output=True, text=True)

        metrics_data = get_sonar_metrics(sonar_project_key, commit_sha)
        if metrics_data:
            save_metrics_to_csv(metrics_data, commit_sha, initial_dir, output_csv)
        
        # Get bug locations and save to CSV
        bugs_data = get_bug_locations(sonar_project_key, commit_sha)
        if bugs_data:
            save_bugs_to_csv(bugs_data, commit_sha, initial_dir, output_csv_bugs)

    except Exception as e:
        print(f"Error executing sonar-scanner: {e}")
    finally:
        os.chdir(initial_dir)


def main(sonar_project_key, output_csv, output_csv_bugs, branch, sonar_sources):
    repo_path = os.path.join(os.getcwd(), 'repos', sonar_project_key)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if not os.path.exists(repo_path):
        print(f"Repo not found in {repo_path}. Cloning...")
        repo = git.Repo.clone_from('REPO_URL', repo_path)
    else:
        repo = git.Repo(repo_path)

    for commit in repo.iter_commits(branch):
        print(f"Processing commit {commit.hexsha}...")
        run_sonar_scanner(commit.hexsha, sonar_project_key, output_csv, output_csv_bugs, branch, sonar_sources)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Executa SonarQube Scanner para cada commit e coleta métricas e bugs.')
    parser.add_argument('--sonar_project_key', required=True, help='Name of the project')
    parser.add_argument('--output_csv', required=True, help='Name of the file to save metrics')
    parser.add_argument('--output_csv_bugs', required=True, help='Name of the file to save bugs')
    parser.add_argument('--branch', required=True, help='Branch name')
    parser.add_argument('--sonar_sources', required=True, help='location of .java files')
    args = parser.parse_args()
    main(args.sonar_project_key, args.output_csv, args.output_csv_bugs, args.branch, args.sonar_sources)
