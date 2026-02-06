from my_instance import *
from my_solution import *
import random

class MyMutation1:
    def __init__(self):
        pass
    
    def do_mutation(self, cur_solution: MySolution) -> None:
        n = cur_solution.problem_instance.n
        solution = cur_solution.solution
        
        # Randomly select two distinct positions to swap
        i, j = random.sample(range(n), 2)
        
        # Swap the cities at these two positions
        solution[i], solution[j] = solution[j], solution[i]

        return cur_solution, (i, j)
    
    def apply(self, cur_solution: MySolution):
        """
        Standard interface: returns only the solution.
        """
        self.do_mutation(cur_solution)
        
       

class MyExtendedMutation1(MyMutation1):
    def __init__(self):
        super().__init__()
        self.improvement_pressure = False
        self.can_worsen = True
        self.repeat_after_fail = True
        self.repeat_after_success = True 
        
    def __str__(self):
        return 'MyMutation1'

    def apply(self, solution):
        super().apply(solution)
        return solution
    def apply_withinfo(self, solution):
        solution, (i, j)=self.do_mutation(solution)
        return solution, ("swap",i, j)

    def __repr__(self):
        return str(self) + '()' 