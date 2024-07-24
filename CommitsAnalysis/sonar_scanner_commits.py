import git
import subprocess
import json
import os

# Configurações

SONAR_SCANNER_CMD = 'sonar-scanner'
SONAR_PROJECT_KEY = 'dubbo'
SONAR_URL = 'http://localhost:9000'  # URL do servidor SonarQube
RESULTS_DIR = 'results_sonar'
REPO_PATH = f'repos/{SONAR_PROJECT_KEY}'
BRANCH = '3.2'
SONAR_BINARIES_PATH = 'target/classes'  # Caminho para os binários compilados, ajustado para um projeto Maven

def run_sonar_scanner(repo, commit_sha):
    initial_dir = os.getcwd()
    os.chdir(REPO_PATH)
    
    try:
        # Check out do commit
        repo.git.checkout(commit_sha)
        
        # Compilar o projeto
        #compile_command = 'mvn clean install -DskipTests'
        #compile_result = subprocess.run(compile_command, capture_output=True, text=True, shell=True)
        
        #if compile_result.returncode != 0:
        #    print(f"Erro ao compilar o projeto no commit {commit_sha}: {compile_result.stderr}")
        #    return
        
        # Configurar o SonarQube Scanner
        sonar_properties = f"""
        sonar.projectKey={SONAR_PROJECT_KEY}
        sonar.sources=.
        sonar.host.url={SONAR_URL}
        sonar.token=squ_9ba02e2dbcd18ef76b9799b766be04369edf3caa
        sonar.sourceEncoding=UTF-8
        sonar.java.binaries={SONAR_BINARIES_PATH}
        """
        
        # Salvar a configuração 
        with open('sonar-project.properties', 'w') as file:
            file.write(sonar_properties)

        print(f"Running sonar-scanner for commit {commit_sha}")
        print(f"Using sonar-project.properties:\n{sonar_properties}")
        print(f"SONAR_SCANNER_CMD: {SONAR_SCANNER_CMD}")
        
        # Executar o SonarQube Scanner
        result = subprocess.run([SONAR_SCANNER_CMD], capture_output=True, text=True, shell=True)
        
        result_file = os.path.join(initial_dir, RESULTS_DIR, f'{commit_sha}.json')
        with open(result_file, 'w') as file:
            json.dump({
                'commit_sha': commit_sha,
                'scanner_output': result.stdout,
                'scanner_error': result.stderr
            }, file)
        
        print(f"Resultados do commit {commit_sha} armazenados em {result_file}")
    
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
        break

if __name__ == "__main__":
    main()
