import git
import subprocess
import json
import os
import csv
import requests

# Configurações
SONAR_SCANNER_CMD = 'sonar-scanner'
SONAR_PROJECT_KEY = 'dubbo'
SONAR_URL = 'http://localhost:9001'
SONAR_TOKEN = 'squ_e98998a79136f5cfe59eb35612b82fbcf2478fa4'
RESULTS_DIR = 'results_sonar'
REPO_PATH = f'repos/{SONAR_PROJECT_KEY}'
BRANCH = '3.3'
SONAR_BINARIES_PATH = 'target/classes'
CSV_FILE = 'sonarqube_metrics.csv'

def get_sonar_metrics():
    metric_keys = 'code_smells,bugs,vulnerabilities,coverage,duplicated_lines_density,ncloc,files,functions,complexity,comment_lines,sqale_index,sqale_debt_ratio'
    measures_url = f'{SONAR_URL}/api/measures/component'
    params = {
        'component': SONAR_PROJECT_KEY,
        'metricKeys': metric_keys
    }
    auth = (SONAR_TOKEN, '')

    response = requests.get(measures_url, params=params, auth=auth)
    print(f"SonarQube API response status: {response.status_code}")
    print(f"SonarQube API response: {response.text}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erro ao obter métricas do SonarQube: {response.text}")
        return None

def save_metrics_to_csv(metrics_data, commit_sha, initial_dir):
    result_file = os.path.join(initial_dir, RESULTS_DIR, CSV_FILE)
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

def run_sonar_scanner(repo, commit_sha):
    initial_dir = os.getcwd()
    os.chdir(REPO_PATH)
    
    try:
        # Check out do commit
        repo.git.checkout(commit_sha)
        
        # Configurar o SonarQube Scanner
        sonar_properties = f"""
        sonar.projectKey={SONAR_PROJECT_KEY}
        sonar.sources=.
        sonar.host.url={SONAR_URL}
        sonar.token={SONAR_TOKEN}
        sonar.login={SONAR_TOKEN}
        sonar.sourceEncoding=UTF-8
        sonar.java.binaries={SONAR_BINARIES_PATH}
        """
        
        # Salvar a configuração 
        with open('sonar-project.properties', 'w') as file:
            file.write(sonar_properties)

        print(f"Running sonar-scanner for commit {commit_sha}")
        print(f"Using sonar-project.properties:\n{sonar_properties}")
        print(f"SONAR_SCANNER_CMD: {SONAR_SCANNER_CMD}")
        
        # Executar o SonarQube Scanner e aguardar sua conclusão
        result = subprocess.run([SONAR_SCANNER_CMD], capture_output=True, text=True, shell=True)
        
        print(f"SonarQube scanner output: {result.stdout}")
        print(f"SonarQube scanner error: {result.stderr}")
        
        result_file = os.path.join(initial_dir, RESULTS_DIR, f'{commit_sha}.json')
        with open(result_file, 'w') as file:
            json.dump({
                'commit_sha': commit_sha,
                'scanner_output': result.stdout,
                'scanner_error': result.stderr
            }, file)
        
        print(f"Resultados do commit {commit_sha} armazenados em {result_file}")
        
        # Obter métricas do SonarQube e salvar em CSV
        metrics_data = get_sonar_metrics()
        if metrics_data:
            save_metrics_to_csv(metrics_data, commit_sha, initial_dir)
    
    except Exception as e:
        print(f"Erro ao processar o commit {commit_sha}: {e}")
    
    finally:
        os.chdir(initial_dir)

def main():
    try:
        print(REPO_PATH)
        repo = git.Repo(REPO_PATH)
    except git.exc.InvalidGitRepositoryError:
        print(f"Erro: O caminho {REPO_PATH} não é um repositório Git válido.")
        return
    except Exception as e:
        print(f"Erro ao abrir o repositório: {e}")
        return

    try:
        commits = list(repo.iter_commits(BRANCH))  
    except Exception as e:
        print(f"Erro ao obter commits: {e}")
        return

    for commit in commits:
        run_sonar_scanner(repo, commit.hexsha)
        # Limpeza
        repo.git.reset('--hard', 'HEAD')
        repo.git.clean('-fd')


if __name__ == "__main__":
    main()
