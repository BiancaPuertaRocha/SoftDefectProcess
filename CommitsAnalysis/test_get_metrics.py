import requests

# Autenticação
token = 'squ_9ba02e2dbcd18ef76b9799b766be04369edf3caa'
auth = (token, '')


# Passo 1: Obter detalhes da tarefa
task_id = 'f8bf2024-8778-4d0d-b88a-18c4dd4f7d11'
sonarqube_url = 'http://localhost:9000'
task_url = f'{sonarqube_url}/api/ce/task?id={task_id}'
task_response = requests.get(task_url, auth=auth)
task_data = task_response.json()

# Verifique se a resposta contém o analysisId
if 'task' in task_data and 'analysisId' in task_data['task']:
    analysis_id = task_data['task']['analysisId']

    # Passo 3: Consultar as métricas do projeto
    component_key = 'dubbo'
    metric_keys = 'code_smells,bugs,vulnerabilities,coverage,duplicated_lines_density,ncloc,files,functions,complexity,comment_lines,sqale_index,sqale_debt_ratio'
    measures_url = f'{sonarqube_url}/api/measures/component?component={component_key}&metricKeys={metric_keys}'
    measures_response = requests.get(measures_url, auth=auth)
    measures_data = measures_response.json()

    # Exibir as métricas
    print(measures_data)
else:
    print("Erro ao obter o analysisId da tarefa:", task_data)