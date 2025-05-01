from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def run_chi_square_random_forest():
    X = df.drop(columns=['failure_prone']) 
    y = df['failure_prone']

    # Normalizando os dados (necessário para o teste Chi-quadrado)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Lista para armazenar a média dos scores de validação cruzada para diferentes k
    scores = []
    selected_features_per_k = []

    # Variação do número de features selecionadas
    for k in range(1, X.shape[1] + 1):  # de 1 até o número total de features
        chi2_selector = SelectKBest(chi2, k=k)
        X_new = chi2_selector.fit_transform(X_scaled, y)
        
        # Treinando o modelo Random Forest com as k melhores características
        rf = RandomForestClassifier(random_state=42)
        
        # Usando validação cruzada para avaliar a performance
        score = cross_val_score(rf, X_new, y, cv=5, scoring='accuracy').mean()
        scores.append(score)
        
        # Obter as características selecionadas para o valor de k atual
        selected_features = X.columns[chi2_selector.get_support()]
        selected_features_per_k.append(selected_features)

    # Encontrar o k que deu o melhor desempenho
    best_k = scores.index(max(scores)) + 1  # o índice é 0-based, por isso adicionamos 1
    best_score = max(scores)

    # Exibir o melhor k e a precisão
    print(f'O melhor número de features (k) é {best_k} com um score de validação de {best_score:.4f}')

    # Exibir as features selecionadas para o melhor k
    print(f'Features selecionadas para o melhor k ({best_k}):')
    print(selected_features_per_k[best_k - 1])

    return selected_features_per_k[best_k - 1]
