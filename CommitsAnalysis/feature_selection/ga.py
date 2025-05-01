import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from deap import base, creator, tools, algorithms
import random


def run_ga_random_forest(df):
    X = df.drop(columns=['failure_prone'])
    y = df['failure_prone']

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    N_FEATURES = X.shape[1]
    POP_SIZE = 20
    N_GEN = 30
    CXPB, MUTPB = 0.5, 0.2 

    # Função de avaliação
    def eval_individual(individual):
        if sum(individual) == 0:
            return 0.0, 
        selected_features = [index for index, bit in enumerate(individual) if bit == 1]
        X_selected = X.iloc[:, selected_features]
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(clf, X_selected, y, cv=5, scoring='f1_macro')  
        return scores.mean(),


    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, N_FEATURES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Executar o algoritmo
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GEN,
                                    stats=stats, halloffame=hof, verbose=True)

    # Melhor subconjunto de atributos
    best_ind = hof[0]
    selected_features = [X.columns[i] for i, bit in enumerate(best_ind) if bit == 1]

    print("\nMelhores atributos selecionados:")
    print(selected_features)

    return selected_features
