import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


class GenericModelBase:
    def prepare_data(self, df, sampler=None, feature_selector=None, threshold=0.5, k=10, balance_first=False):
        """Prepara os dados aplicando tratamento de nulos, balanceamento, seleção de atributos e normalização."""
        X = df.drop('failure_prone', axis=1)
        y = df['failure_prone']

        # Remover valores nulos
        X = X.dropna()
        y = y.loc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)

        # Remover colunas constantes
        X = X.loc[:, X.nunique() > 1]

        if balance_first and sampler:
            X, y = self.apply_sampler(X, y, sampler)

        if feature_selector:
            selected_features = feature_selector(X, y, k=k) if feature_selector.__name__ == 'chi_square_feature_selection' else feature_selector(X, y, threshold=threshold)
            X = X[selected_features]

        if not balance_first and sampler:
            X, y = self.apply_sampler(X, y, sampler)

        # Codificar variáveis categóricas
        X = pd.get_dummies(X, columns=X.select_dtypes(include=['object', 'category']).columns)

        # Normalização
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return train_test_split(X, y, test_size=0.3, random_state=42)

    def apply_sampler(self, X, y, sampler):
        """Aplica o balanceamento nos dados."""
        categorical_columns = (
            [X.columns.get_loc(col) for col in X.select_dtypes(include=['object', 'category']).columns]
            if sampler.__name__ == 'balance_data_with_smotenc' else None
        )
        return sampler(X, y, categorical_features=categorical_columns) if categorical_columns else sampler(X, y)

    def train_and_evaluate(self, model, X_train, X_test, y_train, y_test):
        """Treina o modelo e calcula a AUC."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        return auc_score, y_test, y_pred

    def generic_pipeline(self, df, *args, **kwargs):
        """Deve ser implementado pelas classes filhas."""
        raise NotImplementedError("Este método deve ser implementado na classe filha.")

    # Funções específicas que podem ser usadas em subclasses
    def model_raw(self, df):
        return self.generic_pipeline(df)

    def model_with_sampler(self, df, sampler):
        return self.generic_pipeline(df, sampler=sampler)

    def model_with_feature_selector(self, df, feature_selector):
        return self.generic_pipeline(df, feature_selector=feature_selector)

    def model_sampler_feature_selector(self, df, sampler, feature_selector):
        return self.generic_pipeline(df, sampler=sampler, feature_selector=feature_selector, balance_first=True)
    
    def model_feature_selector_sampler(self, df, sampler, feature_selector):
        return self.generic_pipeline(df, sampler=sampler, feature_selector=feature_selector)
