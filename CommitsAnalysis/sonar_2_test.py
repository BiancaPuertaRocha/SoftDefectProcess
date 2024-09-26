import git
import subprocess
import json
import os
import csv
import requests
import argparse

# Configurações padrão
SONAR_SCANNER_CMD = 'sonar-scanner'
SONAR_URL = 'http://localhost:9001'
SONAR_USER = 'admin'  # Usuário do SonarQube
SONAR_PASS = 'admin'  # Senha do SonarQube
SONAR_TOKEN = None    # O token será gerado automaticamente
RESULTS_DIR = 'results_sonar'


def generate_new_sonar_token():
    """Gera um novo token de autenticação no SonarQube."""
    global SONAR_TOKEN
    token_name = 'sonnar_token_renewed'
    
    print("Gerando um novo token do SonarQube...")
    
    response = requests.post(
        f'{SONAR_URL}/api/user_tokens/generate',
        auth=(SONAR_USER, SONAR_PASS),
        data={'name': token_name}
    )
    
    if response.status_code == 200:
        SONAR_TOKEN = response.json()['token']
        print(f"Novo token gerado: {SONAR_TOKEN}")
    else:
        print(f"Erro ao gerar um novo token: {response.text}")
        return None
    
    return SONAR_TOKEN


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
    
    if response.status_code == 401:  # Falha de autenticação
        print("Token expirado ou inválido. Tentando gerar um novo token...")
        if generate_new_sonar_token():
            # Tentar novamente após gerar novo token
            auth = (SONAR_TOKEN, '')
            response = requests.get(measures_url, params=params, auth=auth)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Erro ao obter métricas do SonarQube após renovar token: {response.text}")
                return None
        else:
            print("Falha ao gerar novo token.")
            return None
    elif response.status_code == 200:
        return response.json()
    else:
        print(f"Erro ao obter métricas do SonarQube: {response.text}")
        return None


def save_metrics_to_csv(metrics_data, commit_sha, initial_dir, csv_file):
    result_file = os.path.join(initial_dir, RESULTS_DIR, csv_file)
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


def build_project(build_tool, skip_tests=True):
    try:
        print(f"Iniciando build do projeto com {build_tool}...")

        if build_tool == 'maven':
            command = ['mvn', 'compile']  # Compilar código sem empacotar
            if skip_tests:
                command.append('-DskipTests')  # Ignorar testes no Maven
            result = subprocess.run(command, capture_output=True, text=True)

            if "Some Enforcer rules have failed" in result.stderr:
                print("Erro no Maven Enforcer Plugin detectado.")
                return False

        elif build_tool == 'gradle':
            command = ['gradle', 'compileJava']  # Compilar código sem empacotar
            if skip_tests:
                command.append('-x')  # Excluir testes no Gradle
                command.append('test')
            result = subprocess.run(command, capture_output=True, text=True)

            if "This version of Shadow supports Gradle 8.3+ only" in result.stderr:
                print("Erro: a versão do plugin Shadow requer Gradle 8.3 ou superior.")
                return False
        else:
            print(f"Ferramenta de build {build_tool} não suportada.")
            return False

        if result.returncode != 0:
            print(f"Erro no build com {build_tool}: {result.stderr}")
            return False

        print(f"Build do projeto realizado com sucesso usando {build_tool}.")
        return True

    except Exception as e:
        print(f"Erro ao executar o build: {e}")
        return False


def run_sonar_scanner(repo, commit_sha, sonar_project_key, sonar_binaries_path, csv_file, build_tool, branch):
    initial_dir = os.getcwd()
    os.chdir(f'repos/{sonar_project_key}')
    
    try:
        # Check out do commit
        repo.git.checkout(commit_sha)

        # Realizar o build do projeto
        build_success = build_project(build_tool)  # Continue independente do sucesso do build
        
        # Configurar o SonarQube Scanner
        sonar_properties = f"""
        sonar.projectKey={sonar_project_key}
        sonar.sources=.
        sonar.host.url={SONAR_URL}
        sonar.token={SONAR_TOKEN}
        sonar.login={SONAR_TOKEN}
        sonar.sourceEncoding=UTF-8
        sonar.java.binaries={sonar_binaries_path}
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
        metrics_data = get_sonar_metrics(sonar_project_key)
        if metrics_data:
            save_metrics_to_csv(metrics_data, commit_sha, initial_dir, csv_file)
    
    except Exception as e:
        print(f"Erro ao processar o commit {commit_sha}: {e}")
    
    finally:
        os.chdir(initial_dir)


def main():
    # Configurando argparse
    parser = argparse.ArgumentParser(description="Rodar o SonarQube Scanner em múltiplos commits.")
    parser.add_argument('--csv_file', required=True, help='Nome do arquivo CSV para salvar as métricas do SonarQube.')
    parser.add_argument('--sonar_project_key', required=True, help='Chave do projeto no SonarQube.')
    parser.add_argument('--branch', required=True, help='Branch a ser analisada.')
    parser.add_argument('--sonar_binaries_path', required=True, help='Caminho para os binários do Java.')
    parser.add_argument('--build_tool', choices=['maven', 'gradle'], required=True, help='Ferramenta de build (maven ou gradle).')

    args = parser.parse_args()

    try:
        repo_path = f'repos/{args.sonar_project_key}'
        print(repo_path)
        repo = git.Repo(repo_path)
    except git.exc.InvalidGitRepositoryError:
        print(f"Erro: O caminho {repo_path} não é um repositório Git válido.")
        return
    except Exception as e:
        print(f"Erro ao abrir o repositório: {e}")
        return

    # Gerar o token do SonarQube no início
    if not generate_new_sonar_token():
        print("Erro ao gerar token inicial do SonarQube. Abortando.")
        return

    try:
        commits = list(repo.iter_commits(args.branch))
        print(f"Commits encontrados: {len(commits)}")

        for commit in commits:
            run_sonar_scanner(repo, commit.hexsha, args.sonar_project_key, args.sonar_binaries_path, args.csv_file, args.build_tool, args.branch)
    
    except Exception as e:
        print(f"Erro durante a execução: {e}")


if __name__ == "__main__":
    main()
