import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from models.random_forest import RandomForestModel
from models.voting_classifier import VotingEnsembleModel
from models.decision_tree import BaggingDecisionTreeModel
from resampling import balance_data_with_smotenc, balance_data_with_adasyn, balance_data_with_undersampling
from feature_selection import fisher_score_feature_selection, chi_square_feature_selection, cfs_feature_selection


def run_experiments(df, model_class, methods, iterations=1): 
    results = []
    print('Running experiments...')

    for method_name, method in methods.items():
        print(f'Running {method_name}...')

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'auc': [], 'f1': []}

        for _ in range(iterations):
            # Chamar o método para obter os resultados
            auc_score, y_test, y_pred = method()

            # Calcular as métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1)
            recall = recall_score(y_test, y_pred, zero_division=1)
            f1 = f1_score(y_test, y_pred, zero_division=1)

            # Adicionar as métricas à lista
            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['auc'].append(auc_score)
            metrics['f1'].append(f1)

        # Calcular a média de cada métrica
        avg_metrics = {metric: sum(values) / len(values) for metric, values in metrics.items()}
        avg_metrics['method'] = method_name  # Adicionar o nome do método para identificação

        results.append(avg_metrics)
    print('Done!')
    return pd.DataFrame(results)


def main():
    # Configurar o argparse
    parser = argparse.ArgumentParser(description='Execute experiments with different models and settings.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing the dataset.')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file where results will be saved.')
    parser.add_argument(
        '--model', type=str, required=True,
        help="Name of the model to use (e.g., 'random_forest')."
    )
    args = parser.parse_args()

    # Ler o arquivo de entrada
    df = pd.read_csv(args.input_file)

    # Configurar os modelos disponíveis
    models = {
        'random_forest': RandomForestModel,
        'decision_tree': BaggingDecisionTreeModel,
        'voting_classifier': VotingEnsembleModel
    }

    # Verificar se o modelo solicitado está disponível
    if args.model not in models:
        print(f"Erro: modelo '{args.model}' não encontrado. Modelos disponíveis: {', '.join(models.keys())}")
        return

    # Instanciar o modelo
    model_class = models[args.model]()  # Passar o dataframe ao instanciar a classe
    
    # Configurar os métodos para o modelo escolhido
    methods = {
        'raw': lambda: model_class.model_raw(df=df),
        'smotenc': lambda: model_class.model_with_sampler(df=df, sampler=balance_data_with_smotenc),
        'adasyn': lambda: model_class.model_with_sampler(df=df, sampler=balance_data_with_adasyn),
        'fisher': lambda: model_class.model_with_feature_selector(df=df, feature_selector=fisher_score_feature_selection),
        'smotenc_fisher': lambda: model_class.model_sampler_feature_selector(
            df=df,
            sampler=balance_data_with_smotenc,
            feature_selector=fisher_score_feature_selection
        ),
        'adasyn_fisher': lambda: model_class.model_sampler_feature_selector(
            df=df,
            sampler=balance_data_with_adasyn,
            feature_selector=fisher_score_feature_selection
        ),
    }

    # Executar os experimentos
    results_df = run_experiments(df, model_class, methods)

    # Salvar os resultados no arquivo de saída
    results_df.to_csv(args.output_file, index=False)

    print(f'Resultados salvos no arquivo: {args.output_file}')


if __name__ == '__main__':
    main()
