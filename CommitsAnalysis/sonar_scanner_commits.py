import git
import subprocess
import json
import os

# Configurações
REPO_PATH = 'repos'
SONAR_SCANNER_CMD = 'sonar-scanner'
SONAR_PROJECT_KEY = 'dubbo'
SONAR_URL = 'http://localhost:9000'  # URL do servidor SonarQube
RESULTS_DIR = 'results_sonar'

# Função para executar o SonarQube Scanner
def run_sonar_scanner(repo, commit_sha):
    os.chdir(REPO_PATH)
    
    try:
        # Check out o commit
        repo.git.checkout(commit_sha)
        
        # Configurar o SonarQube Scanner
        sonar_properties = f"""
        sonar.projectKey={SONAR_PROJECT_KEY}
        sonar.sources=.
        sonar.host.url={SONAR_URL}
        sonar.login=YOUR_SONARQUBE_TOKEN
        """
        
        # Salvar a configuração 
        with open('sonar-project.properties', 'w') as file:
            file.write(sonar_properties)
        
        # Executar o SonarQube Scanner
        result = subprocess.run([SONAR_SCANNER_CMD], capture_output=True, text=True)
        
        result_file = os.path.join(RESULTS_DIR, f'{commit_sha}.json')
        with open(result_file, 'w') as file:
            json.dump({
                'commit_sha': commit_sha,
                'scanner_output': result.stdout,
                'scanner_error': result.stderr
            }, file)
        
        print(f"Resultados do commit {commit_sha} armazenados em {result_file}")
    
    except Exception as e:
        print(f"Erro ao processar o commit {commit_sha}: {e}")

def main():
    try:
        repo = git.Repo(REPO_PATH)
    except git.exc.InvalidGitRepositoryError:
        print(f"Erro: O caminho {REPO_PATH} não é um repositório Git válido.")
        return
    except Exception as e:
        print(f"Erro ao abrir o repositório: {e}")
        return

    try:
        commits = list(repo.iter_commits('main'))  
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
