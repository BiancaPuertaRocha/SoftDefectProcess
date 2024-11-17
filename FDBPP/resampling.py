from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def balance_data_with_smote(X, y, sampling_strategy='auto', random_state=42):
    """
    Aplica o SMOTE para balancear os dados desbalanceados.

    Parâmetros:
    - X: DataFrame ou array com as features.
    - y: Array ou Series com o rótulo alvo.
    - sampling_strategy: Estratégia de amostragem (padrão é 'auto').
    - random_state: Semente para a reprodutibilidade dos resultados.

    Retorna:
    - X_resampled: Features balanceadas.
    - y_resampled: Rótulos balanceados.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

#para dados categoricos
def balance_data_with_smotenc(X, y, categorical_features, sampling_strategy='auto', random_state=42):
    """
    Balanceia os dados desbalanceados, usando SMOTE ou SMOTENC, dependendo do tipo de features.

    Parâmetros:
    - X: DataFrame ou array com as features.
    - y: Array ou Series com o rótulo alvo.
    - categorical_features: Lista de índices das colunas categóricas em X (para SMOTENC).
    - sampling_strategy: Estratégia de amostragem (padrão é 'auto').
    - random_state: Semente para a reprodutibilidade dos resultados.

    Retorna:
    - X_resampled: Features balanceadas.
    - y_resampled: Rótulos balanceados.
    """
    # Transformar rótulos para valores numéricos caso sejam categóricos
    if y.dtype == 'object' or isinstance(y[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Verificar se há colunas categóricas
    if categorical_features:
        # Usar SMOTENC para dados categóricos
        smote = SMOTENC(categorical_features=categorical_features,
                        sampling_strategy=sampling_strategy,
                        random_state=random_state)
    else:
        # Usar SMOTE padrão para dados numéricos
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

    # Aplicar o método escolhido para balancear as classes
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled


def balance_data_with_adasyn(X, y, sampling_strategy='auto', random_state=42, n_neighbors=5):
    """
    Aplica o ADASYN para balancear os dados desbalanceados.

    Parâmetros:
    - X: DataFrame ou array com as features.
    - y: Array ou Series com o rótulo alvo.
    - sampling_strategy: Estratégia de amostragem (padrão é 'auto').
    - random_state: Semente para a reprodutibilidade dos resultados.
    - n_neighbors: Número de vizinhos a serem usados para gerar as amostras sintéticas.

    Retorna:
    - X_resampled: Features balanceadas.
    - y_resampled: Rótulos balanceados.
    """
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state, n_neighbors=n_neighbors)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled



def balance_data_with_undersampling(X, y, sampling_strategy='auto', random_state=42):
    """
    Aplica o Random Under Sampling para balancear os dados desbalanceados.

    Parâmetros:
    - X: DataFrame ou array com as features.
    - y: Array ou Series com o rótulo alvo.
    - sampling_strategy: Proporção desejada de amostras (padrão é 'auto', que balanceia igualmente).
    - random_state: Semente para a reprodutibilidade dos resultados.

    Retorna:
    - X_resampled: Features balanceadas.
    - y_resampled: Rótulos balanceados.
    """
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

