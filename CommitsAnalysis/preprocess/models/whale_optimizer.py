import numpy as np

class WhaleOptimizer:
    """
    Whale Optimization Algorithm (WOA) genérico para otimização contínua.
    """
    def __init__(self, objective_func, bounds, n_whales=10, n_iterations=20, seed=42):
        """
        Args:
            objective_func: função que recebe um vetor (np.ndarray) e retorna um score (float).
            bounds: lista de tuplas (min, max) para cada dimensão do vetor.
            n_whales: número de agentes (baleias).
            n_iterations: número de iterações para o algoritmo.
            seed: semente para aleatoriedade.
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds, dtype=float)
        self.n_whales = n_whales
        self.n_iterations = n_iterations
        self.seed = seed
        self.best_score = -np.inf
        self.best_position = None

    def optimize(self):
        np.random.seed(self.seed)
        dim = len(self.bounds)
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]

        # Inicializa baleias aleatoriamente dentro dos limites
        whales = np.random.uniform(lb, ub, (self.n_whales, dim))
        scores = np.array([self.objective_func(w) for w in whales])
        best_idx = np.argmax(scores)
        self.best_position, self.best_score = whales[best_idx].copy(), scores[best_idx]

        for t in range(self.n_iterations):
            a = 2 - t * (2 / self.n_iterations)  # decresce linearmente de 2 até 0

            for i in range(self.n_whales):
                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r
                p = np.random.rand()

                if p < 0.5:
                    # Encurralamento ou busca exploratória
                    D = np.abs(C * self.best_position - whales[i])
                    whales[i] = self.best_position - A * D
                else:
                    # Movimento em espiral em torno da melhor baleia
                    l = np.random.uniform(-1, 1)
                    D = np.abs(self.best_position - whales[i])
                    whales[i] = D * np.exp(1 * l) * np.cos(2 * np.pi * l) + self.best_position

                # Garantir que as baleias fiquem dentro dos limites
                whales[i] = np.clip(whales[i], lb, ub)

            # Avaliar a população após atualização
            scores = np.array([self.objective_func(w) for w in whales])
            best_idx = np.argmax(scores)
            if scores[best_idx] > self.best_score:
                self.best_position, self.best_score = whales[best_idx].copy(), scores[best_idx]

        return self.best_position, self.best_score
