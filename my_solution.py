from my_instance import *
import numpy as np
import random

class MySolution:
    def __init__(self, instance: MyInstance):
        self.problem_instance = instance
        self.instance = instance
        self.solution = list(range(self.problem_instance.n))
        random.shuffle(self.solution)
    
    def is_feasible(self) -> bool:
        if len(self.solution) != len(set(self.solution)):
            print("Error: The solution contains duplicate cities.")
            return False
        
        if len(self.solution) != self.problem_instance.n:
            print("Error: The solution does not include exactly n cities.")
            return False
        
        return True
    
    def get_objective(self) -> int:
        if not self.is_feasible():
            raise ValueError("The solution is not feasible.")
        
        total_cost = 0
        for i in range(self.problem_instance.n):
            u = self.solution[i]
            v = self.solution[(i + 1) % self.problem_instance.n]
            total_cost += self.problem_instance.cost_matrix[u, v]
        
        return total_cost
    
    def save_to_file(self, output_filename: str) -> None:
        with open(output_filename, 'w') as file:
            file.write(' '.join(str(city + 1) for city in self.solution) + '\n')
    
    def load_from_file(self, input_filename: str) -> None:
        with open(input_filename, 'r') as file:
            line = file.readline().strip()
            self.solution = list(map(int, line.split()))
            self.solution = [city - 1 for city in self.solution]



    def clone(self):
        import copy
        return copy.deepcopy(self)