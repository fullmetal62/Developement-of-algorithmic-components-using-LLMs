from typing import Dict, Type

# import your component classes
from BaseClasses import (
    RandomHillclimber,
    RuinAndRecreate,
    Repeat,
    VariableNeighbourhoodSearch,
    TabuSearchComponent,
    AdaptiveOperatorSelection,
    HybridGeneticComponent,
)
from my_mutation1 import  MyExtendedMutation1
from my_mutation2 import  MyExtendedMutation2


AVAILABLE_COMPONENTS: Dict[str, Type] = {
    "RandomHillClimber": RandomHillclimber,
    "RuinAndRecreate": RuinAndRecreate,
    "Repeat": Repeat,
    "VariableNeighbourhoodSearch": VariableNeighbourhoodSearch,
    "TabuSearchComponent": TabuSearchComponent,
    "AdaptiveOperatorSelection": AdaptiveOperatorSelection,
    "HybridGeneticComponent": HybridGeneticComponent,
}

MUTATION_COMPONENTS: Dict[str, Type] = {
    "Swap 2 cities": MyExtendedMutation1,
    "Reverse-2opt": MyExtendedMutation2,
}