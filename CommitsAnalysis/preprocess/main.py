import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from preprocess.main_preprocessor_runner import MainPreprocessorRunner  

def get_model(model_name):
    if model_name == 'rf':
        return RandomForestClassifier(random_state=42)
    elif model_name == 'voting':
        clf1 = RandomForestClassifier(n_estimators=50, random_state=42)
        clf2 = DecisionTreeClassifier(random_state=42)
        clf3 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        return VotingClassifier(estimators=[('rf1', clf1), ('dt', clf2), ('rf2', clf3)], voting='soft')
    elif model_name == 'bag_rf':
        base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        return BaggingClassifier(estimator=base_estimator, n_estimators=10, random_state=42)
    elif model_name == 'bag_dt':
        base_estimator = DecisionTreeClassifier(random_state=42)
        return BaggingClassifier(estimator=base_estimator, n_estimators=10, random_state=42)
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

    # Carrega os dados
    file_path = args.input
    project_name = get_project_filename(file_path=file_path)
    df = pd.read_csv(file_path)

    # Define o modelo
    model = get_model(args.model)

    # Roda o pipeline
    runner = MainPreprocessorRunner(df, model=model, filename=project_name)
    resultados = runner.run_all()

    # Exibe resultados
    for r in resultados:
        print("\n--- Resultado ---")
        print(f"FS: {r['fs']}, Balancer: {r['balancer']}")
        print(f"AUC: {r['auc']:.4f}, Accuracy: {r['accuracy']:.4f}, Precision: {r['precision']:.4f}, Recall: {r['recall']:.4f}")
        print(f"Features selecionadas: {r['selected_features']}")
        print(f"Melhores parâmetros: {r['best_params']}")

if __name__ == "__main__":
    main()
