import pandas as pd
from sklearn.base import clone

class Preprocessor:
    def __init__(self, fs_strategy, balancer_strategy):
        """
        Classe para aplicar feature selection e balanceamento de dados.

        Args:
            fs_strategy: objeto de seleção de atributos (ex: FisherScoreFeatureSelector).
            balancer_strategy: objeto de balanceamento de dados (ex: ADASYNBalancer).
        """
        self.fs_strategy = fs_strategy
        self.balancer_strategy = balancer_strategy
        self.selected_features = None
        self.best_params = None

    def run(self, df):
        """
        Executa o pré-processamento completo (FS + balanceamento).

        Args:
            df (pd.DataFrame): Dados contendo a variável alvo 'failure_prone'.

        Returns:
            X_resampled (pd.DataFrame): Dados balanceados com features selecionadas.
            y_resampled (pd.Series): Target balanceado.
            dict: Informações adicionais (features selecionadas, melhores parâmetros).
        """
        # 1. Remover colunas irrelevantes, se existirem
        df = df.drop(columns=[col for col in ['sha', 'filename', 'commit_sha', 'commit_date', 'branch_sonar'] if col in df.columns], errors='ignore')

        # 2. Feature selection
        print("Executando seleção de atributos...")
        fs_result = self.fs_strategy.run(df)
        self.selected_features = fs_result['features']
        print(f"Atributos selecionados: {self.selected_features}")

        df_selected = df[self.selected_features + ['failure_prone']]

        # 3. Balanceamento
        print("Aplicando balanceamento...")
        balance_result = self.balancer_strategy.run(df_selected)
        self.best_params = balance_result['best_params']

        # 4. Gerar dados balanceados finais
        X_resampled, y_resampled = self.balancer_strategy._evaluate_params(df_selected, **self.best_params, return_data=True)

        return X_resampled, y_resampled, {
            'features': self.selected_features,
            'best_params': self.best_params
        }
