import git
import subprocess
import json
import os
import csv
import requests
import argparse
import random
import string
from lxml import etree as ET

from utils import generate_new_sonar_token

SONAR_SCANNER_CMD = 'sonar-scanner'
SONAR_URL = 'http://localhost:9001'
SONAR_USER = 'admin'  # SonarQube user
SONAR_PASS = 'admin'  # SonarQube pass
SONAR_TOKEN = None    # Auto generated token
RESULTS_DIR = 'results_sonar'

def get_sonar_metrics(sonar_project_key):
    metric_keys = 'code_smells,bugs,vulnerabilities,coverage,duplicated_lines_density,ncloc,files,functions,complexity,comment_lines,sqale_index,sqale_debt_ratio'
    measures_url = f'{SONAR_URL}/api/measures/component'
    params = {
        'component': sonar_project_key,
        'metricKeys': metric_keys
    }
    auth = (SONAR_TOKEN, '')

    response = requests.get(measures_url, params=params, auth=auth)
    print(f"SonarQube API response status: {response.status_code}")
    print(f"SonarQube API response: {response.text}")
    
    if response.status_code == 401:  # auth failed
        print("Token expired or invalid. Trying to generate a new token...")
        if generate_new_sonar_token(sonar_url=SONAR_URL, sonar_user=SONAR_USER, sonar_pass=SONAR_PASS):
            # Try again after generating new token
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


def run_sonar_scanner(commit_sha, sonar_project_key, output_csv, branch, sonar_sources):
    # Determine repo path
    repo_path = os.path.join(os.getcwd(), 'repos', sonar_project_key)
    
    # Verify repo exists
    if not os.path.exists(repo_path):
        print(f"Repo not found in {repo_path}. Cloning...")
        repo = git.Repo.clone_from('REPO_URL', repo_path)
    else:
        repo = git.Repo(repo_path)

    initial_dir = os.getcwd()
    os.chdir(repo_path)
    
    try:
        # Check out commit
        repo.git.checkout(commit_sha)

        # Configure SonarQube Scanner
        sonar_properties = f"""
        sonar.projectKey={sonar_project_key}
        sonar.host.url={SONAR_URL}
        sonar.login={SONAR_TOKEN}
        sonar.language=java
        sonar.sourceEncoding=UTF-8
        sonar.sources={sonar_sources}
        sonar.language=java
        sonar.java.binaries=
        """
        
        with open('sonar-project.properties', 'w') as f:
            f.write(sonar_properties)
        
        # Execute Sonar Scanner
        result = subprocess.run([SONAR_SCANNER_CMD], capture_output=True, text=True)

        # Save metrics
        metrics_data = get_sonar_metrics(sonar_project_key)
        if metrics_data:
            save_metrics_to_csv(metrics_data, commit_sha, initial_dir, output_csv)


    except Exception as e:
        print(f"Error executing sonar-scanner: {e}")
    finally:
        os.chdir(initial_dir)  # Go back to the initial repo



def main(sonar_project_key, output_csv, branch, sonar_sources):
    # Determine path to the repository
    repo_path = os.path.join(os.getcwd(), 'repos', sonar_project_key)

    # create repo directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Clone repo if not exists
    if not os.path.exists(repo_path):
        print(f"Repo not found in {repo_path}. Cloning...")
        repo = git.Repo.clone_from('REPO_URL', repo_path)
    else:
        repo = git.Repo(repo_path)

    # TO each commit, execute scanner
    for commit in repo.iter_commits(branch):
        print(f"Processing commit {commit.hexsha}...")
        run_sonar_scanner(commit.hexsha, sonar_project_key, output_csv, branch, sonar_sources)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Executa SonarQube Scanner para cada commit.')
    parser.add_argument('--output_csv', required=True, help='Name of the file to save metrics')
    parser.add_argument('--branch', required=True, help='Branch name')
    parser.add_argument('--sonar_sources',  required=True, help='location of .java files')
    args = parser.parse_args()
    main(args.output_csv, args.branch, args.sonar_sources)