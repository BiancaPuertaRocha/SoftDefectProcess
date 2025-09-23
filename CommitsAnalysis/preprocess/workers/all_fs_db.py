import argparse, random, time
import pandas as pd

from preprocess.utils.fs_db_with_model import MainPreprocessorRunner
from preprocess.utils.utils import format_time
from preprocess.utils.classifier_factories import create_rf, create_bagging, create_cart, create_voting

def get_rand():
    """Definição do componente aleatório da rodada"""
    return random.randint(0, 100)

def get_model(model_name, random_state):
    if model_name == 'rf':
        return create_rf(random_state)
    elif model_name == 'voting':
        return create_voting(random_state)
    elif model_name == 'bag_rf':
        return create_bagging(create_rf(random_state), random_state)
    elif model_name == 'bag_dt':
        return create_bagging(create_cart(random_state), random_state)
    else:
        raise ValueError(f"Modelo '{model_name}' não é reconhecido. Use: rf, voting, bag_rf, bag_dt")

def get_project_filename(file_path):
    project_name = file_path.split("/")[-1].split("_")[1]
    return project_name
def main():
    parser = argparse.ArgumentParser(description="Rodar pré-processamento e avaliação de modelos.")
    parser.add_argument('--input', type=str, required=True, help="Caminho do arquivo CSV com os dados.")
    parser.add_argument('--model', type=str, required=True, choices=['rf', 'voting', 'bag_rf', 'bag_dt'],
                        help="Modelo a ser usado: rf, voting, bag_rf, bag_dt")

    args = parser.parse_args()

    file_path = args.input
    project_name = get_project_filename(file_path=file_path)
    df = pd.read_csv(file_path)

    random_state = get_rand()

    model = get_model(args.model, random_state=random_state)

    start = time.time()
    runner = MainPreprocessorRunner(df, model=model, filename=project_name, random_state=random_state)
    resultados = runner.run_all()
    elapsed = time.time() - start

    for r in resultados:
        print("\n--- Resultado ---")
        print(f"FS: {r['fs']}, Balancer: {r['balancer']}")
        print(f"AUC: {r['auc']:.4f}, Accuracy: {r['accuracy']:.4f}, Precision: {r['precision']:.4f}, Recall: {r['recall']:.4f}")
    print(f"Tempo em segundos para execução com otimizacao: {format_time(elapsed)}")

if __name__ == "__main__":
    main()
