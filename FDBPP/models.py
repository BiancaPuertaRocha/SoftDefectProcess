
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from resampling import balance_data_with_smotenc

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
    print(classification_report(y_test_raw, y_pred_raw, zero_division=1))

    # Calcular as probabilidades da classe positiva
    y_pred_proba_raw = model.predict_proba(X_test)[:, 1]

    # Calcular o AUC
    auc_score_raw = roc_auc_score(y_test_raw, y_pred_proba_raw)

    # Exibir o AUC
    print(f"AUC: {auc_score_raw:.4f}")

    return auc_score_raw, y_test_raw, y_pred_raw


def random_forest_smotenc(df):
    X = df.drop('failure_prone', axis=1)
    y = df['failure_prone']

    print('Contagem de itens na classe')
    print(y.value_counts())

    X = X.dropna()
    y = y.loc[X.index]  # Manter os mesmos índices de X
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Identificar colunas categóricas após remover 'commit_id'
    categorical_columns = [X.columns.get_loc(col) for col in X.select_dtypes(include=['object', 'category']).columns]

    # Aplicar SMOTENC usando a função balance_data_with_smotenc
    X_resampled, y_resampled = balance_data_with_smotenc(X, y, categorical_features=categorical_columns)

    print('Contagem de itens na classe - Após SMOTE')
    print(y_resampled.value_counts())

    # Transformar colunas categóricas em numéricas usando One-Hot Encoding
    X_resampled = pd.get_dummies(X_resampled, columns=X.select_dtypes(include=['object', 'category']).columns)

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Treinar o modelo RandomForest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Fazer predições e avaliar
    y_pred_resampled = model.predict(X_test)
    print(classification_report(y_test_resampled, y_pred_resampled))

    y_pred_proba_resampled = model.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva

    # Calcular o AUC
    auc_score_resampled = roc_auc_score(y_test_resampled, y_pred_proba_resampled)

    # Exibir o AUC
    print(f"AUC: {auc_score_resampled:.4f}")

    return auc_score_resampled, y_test_resampled, y_pred_resampled
