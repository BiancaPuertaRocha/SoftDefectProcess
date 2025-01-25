import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from base import GenericModelBase

class BaggingDecisionTreeModel(GenericModelBase):
    def generic_pipeline(self, df, sampler=None, feature_selector=None, threshold=0.5, k=10, balance_first=False):
        """
        Pipeline específico para treinar Bagging com Decision Trees com ou sem balanceamento e seleção de atributos.
        """
        print('Bulding model...')
        X = df.drop('failure_prone', axis=1)
        y = df['failure_prone']

        # Remover valores nulos
        X = X.dropna()
        y = y.loc[X.index]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if balance_first:
            # Aplicar balanceamento primeiro
            if sampler is not None:
                categorical_columns = (
                    [X.columns.get_loc(col) for col in X.select_dtypes(include=['object', 'category']).columns]
                    if sampler.__name__ == 'balance_data_with_smotenc' else None
                )
                X, y = sampler(X, y, categorical_features=categorical_columns) if categorical_columns else sampler(X, y)

            # Aplicar seleção de atributos
            if feature_selector is not None:
                selected_features = feature_selector(X, y, k=k) if feature_selector.__name__ == 'chi_square_feature_selection' else feature_selector(X, y, threshold=threshold)
                X = X[selected_features]
        else:
            # Aplicar seleção de atributos primeiro
            if feature_selector is not None:
                selected_features = feature_selector(X, y, k=k) if feature_selector.__name__ == 'chi_square_feature_selection' else feature_selector(X, y, threshold=threshold)
                X = X[selected_features]

            # Aplicar balanceamento
            if sampler is not None:
                categorical_columns = (
                    [X.columns.get_loc(col) for col in X.select_dtypes(include=['object', 'category']).columns]
                    if sampler.__name__ == 'balance_data_with_smotenc' else None
                )
                X, y = sampler(X, y, categorical_features=categorical_columns) if categorical_columns else sampler(X, y)

        # Codificar variáveis categóricas após o balanceamento e seleção
        X = pd.get_dummies(X, columns=X.select_dtypes(include=['object', 'category']).columns)

        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Treinar o modelo Bagging com Decision Trees
        model = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42), random_state=42, n_estimators=10)
        model.fit(X_train, y_train)

        # Fazer predições e calcular AUC
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

        return auc_score, y_test, y_pred
