#from sklearn.feature_selection import mutual_info_classif
#from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif

import numpy as np
import pandas as pd 

def fisher_score_feature_selection(X, y, threshold=0.5):
    """
    Seleciona os atributos usando o Fisher Score.

    Parâmetros:
    - X: DataFrame com as features.
    - y: Array ou Series com o rótulo alvo.
    - threshold: Limite para o Fisher Score mínimo para seleção.

    Retorna:
    - selected_features: Lista de nomes dos atributos selecionados.
    """
    # Calcular o Fisher Score de cada feature em relação ao alvo
    fisher_scores, _ = f_classif(X, y)
    fisher_scores = pd.Series(fisher_scores, index=X.columns)
    
    # Selecionar features que têm um Fisher Score maior que o limite
    selected_features = fisher_scores[fisher_scores > threshold].index.tolist()

    return selected_features

def cfs_feature_selection(X, y, threshold=0.5):
    """
    Seleciona os atributos usando o método CFS (Correlation-based Feature Selection).

    Parâmetros:
    - X: DataFrame com as features.
    - y: Array ou Series com o rótulo alvo.
    - threshold: Limite para a correlação mínima entre as features e o alvo para seleção.

    Retorna:
    - selected_features: Lista de nomes dos atributos selecionados.
    """
    # Calcular a correlação entre as features e a variável alvo
    corr_with_target = X.apply(lambda col: col.corr(y, method='pearson'))

    # Selecionar features que têm uma correlação maior que o limite com o alvo
    high_corr_features = corr_with_target[abs(corr_with_target) > threshold].index.tolist()

    # Calcular a matriz de correlação entre as features selecionadas
    corr_matrix = X[high_corr_features].corr().abs()

    # Identificar atributos redundantes (altamente correlacionados entre si)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    # Manter apenas os atributos não redundantes
    selected_features = [feature for feature in high_corr_features if feature not in redundant_features]

    return selected_features


def chi_square_feature_selection(X, y, k=10):
    """
    Seleciona os melhores atributos usando o Chi-Square Test.

    Parâmetros:
    - X: DataFrame com as features (deve conter apenas valores numéricos e não-negativos).
    - y: Array ou Series com o rótulo alvo.
    - k: Número de melhores atributos a serem selecionados.

    Retorna:
    - selected_features: Lista de nomes dos atributos selecionados.
    """
    # Transformar y em rótulos numéricos, se necessário
    if y.dtype == 'object' or isinstance(y, pd.Series):
        y = LabelEncoder().fit_transform(y)

    # Aplicar Chi-Square Test para selecionar os k melhores atributos
    chi_selector = SelectKBest(score_func=chi2, k=k)
    X_selected = chi_selector.fit_transform(X, y)

    # Obter os nomes das features selecionadas
    selected_features = X.columns[chi_selector.get_support()].tolist()

    return selected_features