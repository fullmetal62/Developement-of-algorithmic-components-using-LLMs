from my_instance import *
from my_solution import *
import random

class MyMutation2:
    def __init__(self):
        pass
    
    def do_mutation(self, cur_solution: MySolution) -> None:
        n = cur_solution.problem_instance.n
        solution = cur_solution.solution
        
        # Randomly select a subsequence length and a starting index
        length = random.randint(2, n)  # Length of the subsequence to reverse
        start_index = random.randint(0, n - length)
        
        # Reverse the selected subsequence
        end_index = start_index + length
        solution[start_index:end_index] = reversed(solution[start_index:end_index])
        
        return cur_solution, ( start_index, end_index - 1)
    
    def apply(self, cur_solution: MySolution):
        """
        Standard interface: returns only the solution.
        """
        self.do_mutation(cur_solution)

class MyExtendedMutation2(MyMutation2):
    def __init__(self):
        super().__init__()
        self.improvement_pressure = False
        self.can_worsen = True
        self.repeat_after_fail = True
        self.repeat_after_success = True 
        
    def __str__(self):
        return 'MyMutation2'

    def apply(self, solution):
        super().apply(solution)
        return solution
    def apply_withinfo(self, solution):
        solution, (start, end) =self.do_mutation(solution)
        return solution, ("reverse",start, end)   
    def __repr__(self):
        return str(self) + '()' 