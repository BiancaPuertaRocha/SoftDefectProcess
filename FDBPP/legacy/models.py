
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from resampling import balance_data_with_smotenc, balance_data_with_adasyn, balance_data_with_undersampling
from feature_selection import cfs_feature_selection, chi_square_feature_selection, fisher_score_feature_selection

def random_forest_raw(df):
    # Separar X e y
    X = df.drop('failure_prone', axis=1)
    y = df['failure_prone']

    # Transformar colunas categóricas em numéricas usando One-Hot Encoding
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object', 'category']).columns)

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test_raw = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo RandomForest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Fazer predições e avaliar
    y_pred_raw = model.predict(X_test)

    # Calcular as probabilidades da classe positiva
    y_pred_proba_raw = model.predict_proba(X_test)[:, 1]

    # Calcular o AUC
    auc_score_raw = roc_auc_score(y_test_raw, y_pred_proba_raw)

    return auc_score_raw, y_test_raw, y_pred_raw

def random_forest_pipeline(df, sampler=None, feature_selector=None, threshold=0.5, k=10, balance_first=False):
    """
    Pipeline genérico para treinar Random Forest com ou sem balanceamento e seleção de atributos.
    
    Parâmetros:
    - df: DataFrame com as features e o rótulo `failure_prone`.
    - sampler: Função de balanceamento (ex: `balance_data_with_smotenc`, `balance_data_with_adasyn`).
    - feature_selector: Função de seleção de atributos (ex: `fisher_score_feature_selection`, `cfs_feature_selection`, `chi_square_feature_selection`).
    - threshold: Limite para seleção de atributos (usado em Fisher e CFS).
    - k: Número de melhores atributos a serem selecionados (usado em Chi-Square).
    
    Retorna:
    - auc_score: AUC do modelo.
    - y_test: Rótulos reais do conjunto de teste.
    - y_pred: Predições do modelo.
    """
    # Separar X e y
    X = df.drop('failure_prone', axis=1)
    y = df['failure_prone']
    
    # Remover valores nulos
    X = X.dropna()
    y = y.loc[X.index]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # print(sampler.__name__)
    # print(feature_selector.__name__)
    if balance_first:
        # Pré-processamento inicial (categorias ou balanceamento)
        if sampler is not None:
            # Identificar colunas categóricas (apenas para SMOTENC)
            categorical_columns = [
                X.columns.get_loc(col)
                for col in X.select_dtypes(include=['object', 'category']).columns
            ] if sampler.__name__ == 'balance_data_with_smotenc' else None

            # Balancear os dados
            if sampler.__name__ == 'balance_data_with_smotenc':
                X, y = sampler(X, y, categorical_features=categorical_columns)
            else:
                X, y = sampler(X, y)

        # Seleção de atributos
        if feature_selector is not None:
            if feature_selector.__name__ == 'chi_square_feature_selection':
                selected_features = feature_selector(X, y, k=k)
            else:
                selected_features = feature_selector(X, y, threshold=threshold)
            X = X[selected_features]
    else:
        if feature_selector is not None:
            if feature_selector.__name__ == 'chi_square_feature_selection':
                selected_features = feature_selector(X, y, k=k)
            else:
                selected_features = feature_selector(X, y, threshold=threshold)
            X = X[selected_features]

        if sampler is not None:
            # Identificar colunas categóricas (apenas para SMOTENC)
            categorical_columns = [
                X.columns.get_loc(col)
                for col in X.select_dtypes(include=['object', 'category']).columns
            ] if sampler.__name__ == 'balance_data_with_smotenc' else None

            # Balancear os dados
            if sampler.__name__ == 'balance_data_with_smotenc':
                X, y = sampler(X, y, categorical_features=categorical_columns)
            else:
                X, y = sampler(X, y)

    # Codificar variáveis categóricas após o balanceamento e seleção
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object', 'category']).columns)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predições e avaliação
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    return auc_score, y_test, y_pred

# Funções específicas
def random_forest_raw(df):
    return random_forest_pipeline(df)

def random_forest_smotenc(df):
    return random_forest_pipeline(df, sampler=balance_data_with_smotenc)

def random_forest_adasyn(df):
    return random_forest_pipeline(df, sampler=balance_data_with_adasyn)

def random_forest_smotenc_fisher(df):
    return random_forest_pipeline(df, sampler=balance_data_with_smotenc, feature_selector=fisher_score_feature_selection)

def random_forest_adasyn_fisher(df):
    return random_forest_pipeline(df, sampler=balance_data_with_adasyn, feature_selector=fisher_score_feature_selection)

def random_forest_fisher_smotenc(df):
    return random_forest_pipeline(df, feature_selector=fisher_score_feature_selection, sampler=balance_data_with_smotenc)

def random_forest_fisher_adasyn(df):
    return random_forest_pipeline(df, feature_selector=fisher_score_feature_selection, sampler=balance_data_with_adasyn)

def random_forest_smotenc_cfs(df):
    return random_forest_pipeline(df, sampler=balance_data_with_smotenc, feature_selector=cfs_feature_selection)

def random_forest_adasyn_cfs(df):
    return random_forest_pipeline(df, sampler=balance_data_with_adasyn, feature_selector=cfs_feature_selection)

def random_forest_cfs_smotenc(df):
    return random_forest_pipeline(df, feature_selector=cfs_feature_selection, sampler=balance_data_with_smotenc)

def random_forest_cfs_adasyn(df):
    return random_forest_pipeline(df, feature_selector=cfs_feature_selection, sampler=balance_data_with_adasyn)

def random_forest_smotenc_chi(df):
    return random_forest_pipeline(df, sampler=balance_data_with_smotenc, feature_selector=chi_square_feature_selection)

def random_forest_adasyn_chi(df):
    return random_forest_pipeline(df, sampler=balance_data_with_adasyn, feature_selector=chi_square_feature_selection)

def random_forest_chi_smotenc(df):
    return random_forest_pipeline(df, feature_selector=chi_square_feature_selection, sampler=balance_data_with_smotenc)

def random_forest_chi_adasyn(df):
    return random_forest_pipeline(df, feature_selector=chi_square_feature_selection, sampler=balance_data_with_adasyn)