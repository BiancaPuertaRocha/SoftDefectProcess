import git
import subprocess
import json
import os
import csv
import requests
import argparse
import random
import string
import xml.etree.ElementTree as ET

# Configurações padrão
SONAR_SCANNER_CMD = 'sonar-scanner'
SONAR_URL = 'http://localhost:9001'
SONAR_USER = 'admin'  # Usuário do SonarQube
SONAR_PASS = 'admin'  # Senha do SonarQube
SONAR_TOKEN = None    # O token será gerado automaticamente
RESULTS_DIR = 'results_sonar'


def generate_random_token_name(base_name):
    """Gera um nome aleatório baseado em um nome base."""
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    return f"{base_name}_{suffix}"


def generate_new_sonar_token():
    """Gera um novo token de autenticação no SonarQube."""
    global SONAR_TOKEN
    token_name = 'sonnar_token_renewed'
    
    print("Gerando um novo token do SonarQube...")
    
    while True:
        response = requests.post(
            f'{SONAR_URL}/api/user_tokens/generate',
            auth=(SONAR_USER, SONAR_PASS),
            data={'name': token_name}
        )
        
        if response.status_code == 200:
            SONAR_TOKEN = response.json()['token']
            print(f"Novo token gerado: {SONAR_TOKEN}")
            return SONAR_TOKEN
        elif "already exists" in response.text:
            # Gera um novo nome aleatório se o token já existir
            print(f"O token {token_name} já existe. Gerando um novo nome...")
            token_name = generate_random_token_name(token_name)
        else:
            print(f"Erro ao gerar um novo token: {response.text}")
            return None


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


def modify_pom(pom_path):
    """Adiciona a exclusão do plugin Apache RAT no pom.xml."""
    try:
        tree = ET.parse(pom_path)
        root = tree.getroot()

        # Namespace do XML
        ns = {'maven': 'http://maven.apache.org/POM/4.0.0'}

        # Criar a seção de exclusão se não existir
        exclusions = root.find('maven:build/maven:plugins', ns)
        if exclusions is None:
            build = root.find('maven:build', ns)
            if build is None:
                build = ET.SubElement(root, 'build')
            exclusions = ET.SubElement(build, 'plugins')

        # Adicionar a exclusão do Apache RAT
        rat_plugin = ET.SubElement(exclusions, 'plugin')
        group_id = ET.SubElement(rat_plugin, 'groupId')
        group_id.text = 'org.apache.rat'
        artifact_id = ET.SubElement(rat_plugin, 'artifactId')
        artifact_id.text = 'apache-rat-plugin'
        version = ET.SubElement(rat_plugin, 'version')
        version.text = '0.13'

        # Salvar o pom.xml modificado
        tree.write(pom_path, xml_declaration=True, encoding='utf-8')
        print(f"Exclusão do plugin Apache RAT adicionada ao {pom_path}.")
    except Exception as e:
        print(f"Erro ao modificar o pom.xml: {e}")


def build_project(build_tool, skip_tests=True, disable_enforcer=True):
    try:
        print(f"Iniciando build do projeto com {build_tool}...")

        # Modificar o pom.xml para adicionar a exclusão do plugin RAT
        if build_tool == 'maven':
            pom_path = 'pom.xml'
            modify_pom(pom_path)  # Adicione esta linha

            command = ['mvn', 'compile']  # Compilar código sem empacotar
            if skip_tests:
                command.append('-DskipTests')  # Ignorar testes no Maven
            if disable_enforcer:
                command.append('-Denforcer.skip=true')  # Desabilitar Maven Enforcer Plugin
            result = subprocess.run(command, capture_output=True, text=True)

            if "Some Enforcer rules have failed" in result.stderr:
                print("Erro no Maven Enforcer Plugin detectado.")
                print("Tentando compilar novamente com Enforcer Plugin desabilitado...")
                command.append('-Denforcer.skip=true')  # Desabilitar Maven Enforcer Plugin
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Erro no build mesmo com o Enforcer Plugin desabilitado: {result.stderr}")
                    return False
                else:
                    print("Build realizado com sucesso após desabilitar o Maven Enforcer Plugin.")
                    return True

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

        result = subprocess.run([SONAR_SCANNER_CMD], capture_output=True, text=True)

        # Salvar métricas no CSV
        metrics_data = get_sonar_metrics(sonar_project_key)
        if metrics_data:
            save_metrics_to_csv(metrics_data, commit_sha, initial_dir, csv_file)

    except Exception as e:
        print(f"Erro ao executar o sonar-scanner: {e}")
    finally:
        os.chdir(initial_dir)  # Retorna ao diretório inicial


def main(repo_path, sonar_project_key, csv_file, sonar_binaries_path, build_tool, branch):
    # Verifica se o repositório existe
    if not os.path.exists(repo_path):
        print(f"Repositório não encontrado em {repo_path}. Clonando...")
        repo = git.Repo.clone_from('URL_DO_REPOSITORIO', repo_path)
    else:
        repo = git.Repo(repo_path)

    # Cria diretório para resultados do SonarQube
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Para cada commit, executa o SonarQube
    for commit in repo.iter_commits(branch):
        print(f"Processando commit {commit.hexsha}...")
        run_sonar_scanner(repo, commit.hexsha, sonar_project_key, sonar_binaries_path, csv_file, build_tool, branch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Executa SonarQube Scanner para cada commit.')
    parser.add_argument('--repo_path', required=True, help='Caminho do repositório local.')
    parser.add_argument('--sonar_project_key', required=True, help='Chave do projeto no SonarQube.')
    parser.add_argument('--csv_file', required=True, help='Nome do arquivo CSV para salvar as métricas.')
    parser.add_argument('--sonar_binaries_path', required=True, help='Caminho para os binários do SonarQube.')
    parser.add_argument('--build_tool', required=True, choices=['maven', 'gradle'], help='Ferramenta de build a ser utilizada.')
    parser.add_argument('--branch', required=True, help='Nome da branch a ser processada.')

    args = parser.parse_args()
    main(args.repo_path, args.sonar_project_key, args.csv_file, args.sonar_binaries_path, args.build_tool, args.branch)
