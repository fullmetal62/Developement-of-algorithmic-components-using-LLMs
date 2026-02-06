import numpy as np

class MyInstance:
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        self.n = int(lines[0].strip())
        self.cost_matrix = np.zeros((self.n, self.n), dtype=int)
        
        for i in range(1, self.n + 1):
            self.cost_matrix[i - 1] = list(map(int, lines[i].strip().split()))


class MyExtendedInstance(MyInstance):
    def __init__(self, filename):
        self.filename = filename
        super().__init__(filename)

    def __deepcopy__(self, memo):
        return self

    def __str__(self):
        return self.filename

    def __repr__(self):
        return str(self)