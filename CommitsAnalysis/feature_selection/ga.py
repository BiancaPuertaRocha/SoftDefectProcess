import random
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from deap import base, creator, tools, algorithms


class GAFeatureSelector:
    """
    Feature selection using Genetic Algorithm with hyperparameter tuning via Bayesian Optimization (Optuna).
    A customizable classifier is used for evaluation.
    """
    def __init__(self, classifier, n_trials=20, direction="maximize", sampler=None):
        self.classifier = classifier
        self.n_trials = n_trials
        self.direction = direction
        self.sampler = sampler
        self.study = None
        self.best_params = None
        self.best_score = None

    def _run_ga(self, df, pop_size, n_gen, cxpb, mutpb):
        X = df.drop(columns=['failure_prone'])
        y = df['failure_prone']

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        N_FEATURES = X.shape[1]

        def eval_individual(individual):
            if sum(individual) == 0:
                return 0.0,
            selected_features = [i for i, bit in enumerate(individual) if bit == 1]
            X_selected = X.iloc[:, selected_features]
            clf = self.classifier
            scores = cross_val_score(clf, X_selected, y, cv=5, scoring='f1_macro')
            return scores.mean(),

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, N_FEATURES)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
                                     stats=stats, halloffame=hof, verbose=False)

        best_ind = hof[0]
        selected_features = [X.columns[i] for i, bit in enumerate(best_ind) if bit == 1]
        X_best = X[selected_features]
        final_score = cross_val_score(self.classifier, X_best, y, cv=5, scoring='f1_macro').mean()

        return final_score

    def _objective(self, trial):
        pop_size = trial.suggest_int("pop_size", 10, 50)
        n_gen = trial.suggest_int("n_gen", 10, 50)
        cxpb = trial.suggest_float("cxpb", 0.4, 0.9)
        mutpb = trial.suggest_float("mutpb", 0.01, 0.3)

        try:
            score = self._run_ga(self.df, pop_size, n_gen, cxpb, mutpb)
        except Exception:
            return 0.0
        return score

    def run(self, df):
        self.df = df
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print("\nBest hyperparameters found:")
        print(self.best_params)
        print("Best F1-macro:", self.best_score)

        return self.best_params
