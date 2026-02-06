import numpy as np
import random
from collections import deque
from my_mutation1 import MyExtendedMutation1
from my_mutation2 import MyExtendedMutation2
from neighbourhood import *
from typing import List


class ComponentBase:
    def apply(self, solution):
        raise NotImplementedError()

    def __repr__(self):
        return str(self)


class RandomHillclimber(ComponentBase):
    def __init__(self, mutation_component, iterations):
        self.mutation_component = mutation_component
        self.iterations = iterations
        self.improvement_pressure = True
        self.can_worsen = False
        self.repeat_after_success = True
        self.repeat_after_fail = True

    def apply(self, solution):
        saved = solution.clone()
        obj = solution.get_objective()
        for _ in range(self.iterations):
            # if not solution.is_feasible():
            #     raise Exception()

            solution = self.mutation_component.apply(solution)

            # if not solution.is_feasible():
                # raise Exception()

            new_obj = solution.get_objective()
            if new_obj < obj:
                saved = solution.clone()
                obj = new_obj
            else:
                solution = saved.clone()
                # if not solution.is_feasible():
                #     raise Exception()

        return solution

    def __str__(self):
        return f'HC({self.mutation_component}, {self.iterations})'

    def __str__(self):
        return f'RandomHillclimber({repr(self.mutation_component)}, {self.iterations})'


class Repeat(ComponentBase):
    def __init__(self, component, iterations):
        self.component = component
        self.iterations = iterations
        if isinstance(self.component, ComponentBase):  
            self.improvement_pressure = component.improvement_pressure
            self.can_worsen = component.can_worsen
        self.repeat_after_success = True
        self.repeat_after_fail = True

    
    def apply(self, solution):
        for _ in range(self.iterations):
            solution = self.component.apply(solution)
                
        return solution

    def __str__(self):
        return f'Rep({self.component}, {self.iterations})'

    def __repr__(self):
        return f'Repeat({repr(self.component)}, {self.iterations})'
        
    def apply(self, solution):
        for _ in range(self.iterations):
            
            solution = self.component.apply(solution)
        return solution

    def __str__(self):
        return f'Rep({self.component}, {self.iterations})'

    def __repr__(self):
        return f'Repeat({repr(self.component)}, {self.iterations})'


class RuinAndRecreate(ComponentBase):
    def __init__(self, solution_class):
        self.solution_class = solution_class
        self.improvement_pressure = False
        self.can_worsen = True
        self.repeat_after_success = True
        self.repeat_after_fail = True

    def apply(self, solution):
        """
        Very simple ruin-and-recreate:
        - clone the current solution
        - randomly shuffle the tour
        - return the new solution

        This avoids any file I/O or path handling, so no 'stat' errors.
        """
        import random
        new_sol = solution.clone()
        random.shuffle(new_sol.solution)
        return new_sol
    
class VariableNeighbourhoodSearch(ComponentBase):
 
#    Basic VNS with shake + VND (first-improvement) for local descent.
    is_controller = False

    def __init__(self,
                 neighbourhoods: List[Neighbourhood],
                 kmax: int | None = None,
                 shake_strength: int = 1,
                 max_outer_loops: int = 20):
        assert neighbourhoods, "Provide at least one neighbourhood"
        self.neighbourhoods = neighbourhoods
        self.kmax =  len(neighbourhoods)
        self.shake_strength = max(1, int(shake_strength))
        self.max_outer_loops = max(1, int(max_outer_loops))

        self.improvement_pressure = True
        self.can_worsen = False
        self.repeat_after_success = True
        self.repeat_after_fail = True

    #  VND (local descent across all neighborhoods)
    def _vnd(self, x):
        k = 0
        while k < len(self.neighbourhoods):
            nb = self.neighbourhoods[k]
            new_solution, improved = nb.first_improvement(x)
            if improved:
                x = new_solution
                k = 0   # restart from first neighborhood on improvement
            else:
                k += 1
        return x

    def apply(self, solution):
        current_solution = solution.clone()
        cur_cost = current_solution.get_objective()

        outer = 0
        k = 0
        while k < self.kmax and outer < self.max_outer_loops:
            outer += 1
            nb = self.neighbourhoods[k]

            # SHAKE in N_k 
            new_solution = current_solution.clone()
            for _ in range(self.shake_strength):
                new_solution = nb.random_move(new_solution)

            #LOCAL DESCENT (VND over all neighborhoods)
            latest_solution = self._vnd(new_solution)
            new_cost = latest_solution.get_objective()

            if new_cost  < cur_cost:
                current_solution, cur_cost = latest_solution, new_cost
                k = 0  # improvement: restart from N1
            else:
                k += 1
        
        
        return current_solution

    def __str__(self):
        names = ",".join(nb.name for nb in self.neighbourhoods)
        return f"VNS(shake={self.shake_strength}; [{names}])"


class TabuSearchComponent(ComponentBase):
    def __init__(self, mutation_component, tabu_tenure=5, max_attempts=50):
        self.mutation_component = mutation_component
        self.tabu_list = deque(maxlen=tabu_tenure)  # FIFO fixed-size memory
        self.tabu_tenure = tabu_tenure
        self.max_attempts = max_attempts

    def apply(self, solution):
        best_cost= solution.get_objective()
        for _ in range(10): # try 10 times to find a non-tabu move
            new_solution, move = self.mutation_component.apply_withinfo(solution)
            new_cost= new_solution.get_objective()
            if new_cost>best_cost or move in self.tabu_list:
                continue
        self.tabu_list.append(move)
        return new_solution #, True, move
        

class AdaptiveOperatorSelection(ComponentBase):
    #controllers = {"Repeat", "AdaptiveOperatorSelection"}
    def __init__(self, components_dict, solution_class, instance_data):
        
        self.components_dict = components_dict
        self.solution_class = solution_class
        self.instance_data = instance_data

        # Which components are we allowed to choose?
        # Exclude AOS itself to avoid recursion.

        self.component_names = [name for name in components_dict.keys()]

        self.weights = [1.0] * len(self.components_dict)

        # CMCS-style flags
        self.improvement_pressure = True
        self.can_worsen = True
        self.repeat_after_success = True
        self.repeat_after_fail = True

    # --- helper to instantiate a component with default parameters ---
    def build_component_aos(self, name):
        ComponentClass = self.components_dict[name]

        # You can tune these defaults as you like
        if name == "RandomHillClimber":
            mut = random.choice([MyExtendedMutation1(), MyExtendedMutation2()])
            return ComponentClass(mut, iterations=1000)

        

        elif name == "RuinAndRecreate":
            return ComponentClass(self.solution_class)

        elif name == "VariableNeighbourhoodSearch":
            # Only 2 neighborhoods as requested
            nbs = [TwoOptNeighbourhood(), SwapNeighbourhood()]
            shake =  random.randint(1,5)      # int(llm_args.get("shake_strength", 1))
            loops = random.randint(1,5)    # int(llm_args.get("max_outer_loops", 10)) 
            return ComponentClass(
            neighbourhoods=nbs,
            kmax=len(nbs),
            shake_strength=shake,
            max_outer_loops=loops
            )

        elif name == "TabuSearchComponent":
            mut = random.choice([MyExtendedMutation1(), MyExtendedMutation2()])
            tabu_tenure=7
            max_attempts=50
            return ComponentClass(mut, tabu_tenure, max_attempts)

        elif name == "HybridGeneticComponent":
            mut = random.choice([MyExtendedMutation1(), MyExtendedMutation2()])
            hgc = ComponentClass(self.solution_class, mut)
            hgc.initialize_population(self.instance_data)
            return hgc

        else:
            # If you add more components later, handle them here
            raise ValueError(f"[AOS] Unknown component name: {name}")

    
    
    def apply(self, solution):
        """
        Choose a component name according to weights, build that component,
        apply it to the current solution, update the weight, and
        return ONLY the new solution (MySolution).
        """
        if not self.component_names:
            return solution

        old_cost = solution.get_objective()

        # --- choose index by weights ---
        total_weight = sum(self.weights)
        if total_weight <= 0:
            idx = random.randrange(len(self.component_names))
        else:
            probs = [w / total_weight for w in self.weights]
            idx = random.choices(range(len(self.component_names)), weights=probs, k=1)[0]
        
        name = self.component_names[idx]
        
        component = self.build_component_aos(name)

        try:
            new_solution = component.apply(solution)
        except Exception as e:
            print(f"AOS DEBUG, Selected={name} type={type(component).__name__}")
            raise
        new_cost = new_solution.get_objective()

        success = new_cost < old_cost

        #update weight for that component
        if success:
            self.weights[idx] *= 1.05
        else:
            self.weights[idx] *= 0.95

        return new_solution

    def bootstrap_from_history(self, log_path):
        
        import os
        import pandas as pd
        import numpy as np
        from collections import defaultdict

        if not os.path.exists(log_path):
            return

        try:
            df = pd.read_csv(log_path)
        except Exception as e:
            print("[AOS bootstrap] Failed to read log:", e)
            return

        if "Component" not in df.columns or "Delta" not in df.columns:
            print("[AOS bootstrap] Log missing 'Component'/'Delta' columns; skipping.")
            return

        score_history = defaultdict(list)

        for _, row in df.iterrows():
            comp = row["Component"]
            delta = row["Delta"]
            delta=-delta
            score_history[comp].append(delta)

        # Map average Delta back onto our component_names
        for i, name in enumerate(self.component_names):
            if name in score_history and score_history[name]:
                avg_score = float(np.mean(score_history[name]))
                self.weights[i] = max(avg_score, 0.1)  # keep positive min weight

        print("Bootstrapped AOS weights:", self.weights)


class HybridGeneticComponent(ComponentBase):
    def __init__(self, solution_class, mutation_component, population_size=5):
        self.solution_class = solution_class
        self.mutation_component = mutation_component
        self.population_size = population_size
        self.population = []
        self.improvement_pressure = True
        self.can_worsen = True
        self.repeat_after_success = True
        self.repeat_after_fail = True

    def initialize_population(self, instance):
        self.population = [self.solution_class(instance) for _ in range(self.population_size)]

    def crossover(self, parent1, parent2):
        # Order Crossover (OX) style
        size = len(parent1.solution)
        start, end = sorted(random.sample(range(size), 2))
        child_solution = [None] * size
        child_solution[start:end+1] = parent1.solution[start:end+1]
        fill_pos = (end + 1) % size

        for city in parent2.solution:
            if city not in child_solution:
                while child_solution[fill_pos] is not None:
                    fill_pos = (fill_pos + 1) % size
                child_solution[fill_pos] = city

        child = parent1.clone()
        child.solution = child_solution
        return child

    def apply(self, current_solution):
        if len(self.population) < self.population_size:
            self.population.append(current_solution.clone())
            return current_solution
        # Selection (tournament selection)
        parents = random.sample(self.population, 2)
        child = self.crossover(parents[0], parents[1])

        # Apply mutation
        child = self.mutation_component.apply(child)

        # Replacement if improved
        child_cost = child.get_objective()
        worst = max(self.population, key=lambda s: s.get_objective())
        if child_cost < worst.get_objective():
            self.population.remove(worst)
            self.population.append(child)
            return child
        else:
            return current_solution

