from __future__ import annotations
import random
from my_solution import MySolution
class Neighbourhood:
    """Abstract neighborhood for TSP tours."""
    name: str

    def first_improvement(self, solution):
        """Return (new_solution, improved: bool)."""
        raise NotImplementedError

    def random_move(self, solution):
        """Apply a random move from this neighborhood and return the new solution."""
        raise NotImplementedError

    # --- helpers (simple and readable) ---
    def _dist(self, solution, i, j):
        inst = getattr(solution, "problem_instance", None)
        if inst is None:
            raise AttributeError("Solution must carry a .problem_instance reference with distances")
        
        if hasattr(inst, "cost_matrix"):
            return inst.cost_matrix[i][j]
        raise AttributeError("Instance must provide cost_matrix[][]")

    def _tour_cost(self, solution, tour):
        n = len(tour)
        total = 0
        for k in range(n):
            a = tour[k]
            b = tour[(k + 1) % n]
            total += self._dist(solution, a, b)
        return total

class TwoOptNeighbourhood(Neighbourhood):
    def __init__(self):
        self.name = "2-opt"

    def first_improvement(self, solution):
        tour = solution.solution
        n = len(tour)
        base_cost = solution.get_objective()

        for i in range(n - 1):
            a = tour[i]
            b = tour[(i + 1) % n]
            # j starts from i+2 to avoid adjacent edges; avoid closing edge overlap
            for j in range(i + 2, n - (0 if i > 0 else 1)):
                c = tour[j]
                d = tour[(j + 1) % n]

                # cost delta (simple & clear: full recompute only if promising)
                # Here we use delta formula; if you want super-simple, recompute full after building new tour.
                before = self._dist(solution, a, b) + self._dist(solution, c, d)
                after  = self._dist(solution, a, c) + self._dist(solution, b, d)
                if after  < before:  
                    new_tour = tour[:i + 1] + list(reversed(tour[i + 1:j + 1])) + tour[j + 1:]
                    new_sol = solution.clone()
                    new_sol.solution = new_tour
                    # no need to recompute here; caller can check objective later
                    return new_sol, True

        return solution, False

    def random_move(self, solution):
        tour = solution.solution[:]
        n = len(tour)
        if n < 4:
            return solution.clone()
        # pick two non-adjacent cut points
        i = random.randrange(0, n - 3)
        j = random.randrange(i + 2, n - (0 if i > 0 else 1))
        new_tour = tour[:i + 1] + list(reversed(tour[i + 1:j + 1])) + tour[j + 1:]
        new_sol = solution.clone()
        new_sol.solution = new_tour
        return new_sol
    
class SwapNeighbourhood(Neighbourhood):
    def __init__(self):
        self.name = "Swap"

    def first_improvement(self, solution):
        tour = solution.solution
        n = len(tour)
        base_cost = solution.get_objective()

        # simple & readable: full-cost recompute after swap (fine for medium n)
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_tour = tour[:]
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                new_cost = self._tour_cost(solution, new_tour)
                if new_cost + 1e-12 < base_cost:
                    new_sol = solution.clone()
                    new_sol.solution = new_tour
                    return new_sol, True
        return solution, False

    def random_move(self, solution):
        tour = solution.solution[:]
        n = len(tour)
        if n < 2:
            return solution.clone()
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
        new_sol = solution.clone()
        new_sol.solution = tour
        return new_sol